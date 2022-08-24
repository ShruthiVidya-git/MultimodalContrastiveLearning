"""

Adopted from Gloria repo https://github.com/marshuang80/gloria

Below code describes the model vit GLoria along with the vision transfomers as image encoders and bert as text encoder and loss functions

"""

#mount drive
from google.colab import drive
drive.mount('/content/drive/')
# import libraries
import tensorflow as tfa
import sys
import torch.nn as nn
import torch
import torchvision
from torchvision import models as models_2d
from sklearn import metrics
from tqdm import tqdm
import pickle, os, pathlib, re, numpy as np, pandas as pd, glob, gc
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import glob, random, os, warnings
from torch.autograd import Variable

"""
Class vit transformers defines image and text encoder an djust uses the global loss function
"""

class VisionTransformer( nn.Module):
    
    def __init__(self):
        super(VisionTransformer,self).__init__()
      
        # self.txt_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)

        # for param in self.txt_model.parameters():
        #     param.requires_grad = False
         
        # specificatoins for vit
         
        self.patch_size = 32
        self.num_channels = 3
        self.num_heads = 8
        self.embed_dim = 768
        self.hidden_dim = 512
        self.num_patches = (256 // self.patch_size) ** 2
        self.dropout= 0.1
        self.num_layers = 6
        
        # Layers/Networks
        self.input_layer = nn.Linear(self.num_channels*(self.patch_size**2), self.embed_dim)

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads,
                                          dropout=self.dropout)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.Dropout(self.dropout))
       
        #print([[self.attentionBlock[i] for i in range(len(self.attentionBlock))] for _ in range(self.num_layers)])
        #self.transformer = nn.Sequential(*[ self.attentionBlock for _ in range(self.num_layers)])

            
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim)
           # nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(self.dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+self.num_patches,self.embed_dim))

        # from attentionBlock
 
    #image to patch
    def img_to_patch(self, x, patch_size = 32, flatten_channels=True):

       B, C, H, W = x.shape
       x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
       x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
       x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
       if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
       return x



    def encode(self, x):
        # Preprocess input
        x = self.img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]
        x1 = x
        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x)) 
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        '''
        x = self.al1(x)
        x = self.al2(x)
        x = self.al3(x)
        x = self.al4(x)
        x = self.al5(x)
        x = self.al6(x)
        '''
        cls = x[0]
        return cls
        
        
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim) 
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    # function global loss
    def global_loss(self,cnn_code, rnn_code, eps=1e-8, temp3=10.0):

        batch_size = cnn_code.shape[0]
        #print(batch_size)
        labels = Variable(torch.LongTensor(range(batch_size)))
        #print(labels)'
      
        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            rnn_code = rnn_code.unsqueeze(0)
      
        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

        #print(cnn_code_norm.shape)
        #print(rnn_code_norm.shape)

        scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
        scores0 = scores0 / norm0.clamp(min=eps) * temp3

        # --> batch_size x batch_size
        scores0 = scores0.squeeze(0)

        scores1 = scores0.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
        return loss0, loss1
      
      
    def _calc_global_loss(self,img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g)
        return g_loss0, g_loss1




    def calc_loss(self,img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # weighted loss
        loss = 0
        loss_g = 0
        #loss_l = 0
        #local_loss_weight = 1
        global_loss_weight = 1
        #loss += (l_loss0 + l_loss1) * local_loss_weight
        loss += (g_loss0 + g_loss1) * global_loss_weight
        #loss_l += l_loss0 + l_loss1
        return loss
    
    #similarity global function 
    def get_global_similarities( self, img_emb_g, text_emb_g):
        #img_emb_g = img_emb_g.detach().cpu().numpy()
        #text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities
