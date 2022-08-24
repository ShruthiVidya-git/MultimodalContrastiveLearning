"""
Adopted from Gloria Repository

This file consists of the similarities computation used durin inference to identify the closest class based on prompts

"""

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, cv2, numpy as np, pickle, os, re, tqdm, random, sys
from sklearn import metrics

def get_similarities(img_emb_g, text_emb_g, img_emb_l, text_emb_l, cap_lens):
    # get similarities
    global_similarities = get_global_similarities(img_emb_g, text_emb_g)
    local_similarities = get_local_similarities(
        img_emb_l, text_emb_l, cap_lens
    )
    similarity = local_similarities + (global_similarities*10)
    #similarities = normalize(similarity)
    return similarity.detach().cpu().numpy()

def normalize(similarities):
    return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0))

def get_global_similarities( img_emb_g, text_emb_g):
    img_emb_g = img_emb_g.detach().cpu().numpy()
    text_emb_g = text_emb_g.detach().cpu().numpy()
    global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
    global_similarities = torch.Tensor(global_similarities)
    return global_similarities.squeeze()

def get_local_similarities(img_emb_l, text_emb_l, cap_lens):

    # at inference we pass the test examples one at a time
    words_num = int(cap_lens)
    context = img_emb_l 

    word = (
                text_emb_l.squeeze()[ :, 1 : words_num + 1].unsqueeze(0).contiguous()
            ) # 1,512, 768

    context = img_emb_l

    weiContext, attn = attention_fn(
                word, context, 4.0
            )  # [1, 512, 10], [1, 10, 16, 16]

    word = word.transpose(1, 2).contiguous()  # [1, 10, 512]
    weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

    word = word.view(words_num, -1)  # [1200, 768]
    weiContext = weiContext.view( words_num, -1)  # [1200, 768]
    
    row_sim = cosine_similarity(word, weiContext)
    row_sim = row_sim.view(1, words_num)  # [48, 25]

    row_sim.mul_(5.0).exp_()
    row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

    row_sim = torch.log(row_sim)
    return torch.Tensor(row_sim).squeeze()

def attention_fn(query, context, temp1):
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

def cosine_similarity( x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim."""
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()