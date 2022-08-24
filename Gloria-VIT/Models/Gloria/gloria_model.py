
"""
Adopted from gloria repo https://github.com/marshuang80/gloria

Below code describes the model GLoria along with the encoders and loss functions

"""

# importing libraries


import torch.nn as nn
import torch
import torchvision
from torchvision import models as models_2d
from sklearn import metrics
from torch.autograd import Variable
from torchsummary import summary

# Mount Drive
from google.colab import drive
drive.mount('/content/drive/')


# class Gloria

class Gloria(nn.Module):
    def __init__(self):
        super(Gloria, self).__init__()
        
        self.output_dim = 1024
        # model_function = getattr(model,cfg.model.vision.model_name)
        # self.txt_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)

        # for param in self.txt_model.parameters():
        #     param.requires_grad = False
        #interm dimensions dimension 1024
        
        self.interm_feature_dim = 1024
        
        #importing pretrained resent50 

        self.model = models_2d.resnet50(weights="IMAGENET1K_V2")

        self.feature_dims = self.model.fc.in_features
        #self.model.fc = self.forward(self.model)
        
        ## global and local embedder 
        # global enbedder -> linear in nature
        # local embedder -> 2d vector
 
        self.global_embedder = nn.Linear(self.feature_dims, self.output_dim)
        self.local_embedder = nn.Conv2d(self.interm_feature_dim, self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            ) 
            
            # the learning representaion layer for image embeddings for global features
            
        self.layers_image_global = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, 1000),
                nn.ReLU(),
                nn.Linear(1000,920),
                nn.ReLU(),
                nn.Linear(920,840),
                nn.ReLU(),
                nn.Linear(840,768),
                nn.ReLU(),
                nn.ReLU(),
                nn.Linear(768,768),
                nn.ReLU()
            )
            
            # the learning representaion layer for image embeddings for local features

        self.layers_image_local = nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 1),
            )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) 
        self.up = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True) 

    def forward(self, x):
        return x


    # getting local/global features from input tensor 
    def get_local_global_features(self,input_array):
        x = self.up(input_array)
        
        x = self.model.conv1(x.float())  
        
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)  
        x = self.model.layer2(x)  
        x = self.model.layer3(x)  
        local_features = x
        x = self.model.layer4(x)  
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        global_emb = self.global_embedder(x)
        local_emb = self.local_embedder(local_features)
        global_emb = self.layers_image_global(global_emb)
        local_emb = self.layers_image_local(local_emb)
        return global_emb, local_emb 
        
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


    def attention_fn(self, query, context, temp1):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        batch_size, queryL = query.size(0), query.size(2)
        ih, iw = context.size(2), context.size(3)
        sourceL = ih * iw

        # --> batch x sourceL x ndf
        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)
    
        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        weightedContext = torch.bmm(context, attnT)

        return weightedContext, attn.view(batch_size, -1, ih, iw)
        
        #global loss calculation
         
    def global_loss(self,cnn_code, rnn_code, eps=1e-8, temp3=10.0):

        batch_size = cnn_code.shape[0]
        #print(batch_size)
        labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)
        #print(labels)
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)

        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * temp3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze()

        scores1 = scores0.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        return loss0, loss1
 
       # local loss calculation
 
    def local_loss(self,img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):
        """
        CrossEntropyLoss computed using cosine similarities and
        get the local attention weights
        """
        batch_size = img_features.shape[0]

        att_maps = []
        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):

        # Get the i-th text description
            words_num = cap_lens[i]  # get each word count of each sentence
            word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 9]
            word = word.repeat(batch_size, 1, 1)  # [25, 768, 9]
            context = img_features  # [25, 768, 16, 16]

            weiContext, attn = self.attention_fn(
                word, context, temp1
                )  # [25, 768, 9], [25, 9, 16, 16]

            att_maps.append(
            attn[i].unsqueeze(0).contiguous()
                )  # add attention for curr index  [25, 16, 16]
            word = word.transpose(1, 2).contiguous()  # [25, 9, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [25, 9, 768]

            word = word.view(batch_size * words_num, -1)  # [225, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [225, 768]

            row_sim = self.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [25, 9]

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [25, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [25, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [25, 25]

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        return loss0, loss1, att_maps 

    def _calc_local_loss(self,img_emb_l, text_emb_l, sents):
        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
                        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
                img_emb_l,
                text_emb_l,
                cap_lens
                )
        return l_loss0, l_loss1, attn_maps    

    def _calc_global_loss(self,img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g)
        return g_loss0, g_loss1

    def calc_loss(self,img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents):
        l_loss0, l_loss1, attn_maps =self._calc_local_loss(
                img_emb_l, text_emb_l, sents
                )
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # weighted loss
        loss = 0
        loss_g = 0
        loss_l = 0
        local_loss_weight = 1
        global_loss_weight = 1
        loss += (l_loss0 + l_loss1) * local_loss_weight
        loss += (g_loss0 + g_loss1) * global_loss_weight
        loss_g += g_loss0 + g_loss1
        loss_l += l_loss0 + l_loss1
        return loss, attn_maps, loss_g, loss_l

    def get_global_similarities( self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, sents):

        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            cap_lens = len([w for w in sents[i] if not w.startswith("[")])

            words_num = cap_lens
            word = (
                    text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
                    )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [25, 768, 9]
            context = img_emb_l  # [25, 768, 16, 16]

            weiContext, attn = self.attention_fn(
            word, context, 4.0
                )  # [48, 768, 25], [25, 9, 16, 16]

            word = word.transpose(1, 2).contiguous()  # [25, 9, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [25, 9, 768]

            word = word.view(batch_size * words_num, -1)  # [225, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [225, 768]
        
            row_sim = self.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [25, 9]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [25, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

