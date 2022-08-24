"""

This contains the training files for VIT+Bert Model
   
"""

from google.colab import drive
drive.mount('/content/drive/')

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

# Import directory
import sys
sys.path.insert(0,'/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Models/')
import vit
from vit import VisionTransformer

# Import directory
sys.path.insert(0,'/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Datasets/')
import data_loader
from data_loader import DataLoader

def tensor_reshape(img_tensor):

    if img_tensor.shape[2] != 256:
        tensor_out = nn.functional.pad(input=img_tensor, pad=(0, 0, 1,0) , mode='constant', value=0)
        return tensor_out

    elif img_tensor.shape[3] != 256:
        tensor_out = nn.functional.pad(input=img_tensor, pad=(0, 1, 0,0) , mode='constant', value=0)
        return tensor_out

    return img_tensor

def _get_batch_tensors(df):
    """
    Get batches of image and text tensors
    """
    img_paths = list(df['img_emd'])
    txt_paths = list(df['txt_emd'])

    img_tensors = []
    text_global_embeddings = []
    pdf_hash = []

    for idx in range(len(df)):
        # read batch of 25 image tensors
        with open(img_paths[idx], 'rb') as f:
            img_tensor = pickle.load(f)
        tensor_out = tensor_reshape(img_tensor)

        img_tensors.append(tensor_out)

        # read batch of 25 txt embeddings
        with open(txt_paths[idx], 'rb') as f:
            txt_tensor = pickle.load(f)
        text_global_embeddings.append(txt_tensor['text_global_embeddings'][0])
        pdf_hash.append(txt_tensor['pdf_hash'])

    txt_g = torch.stack(text_global_embeddings)

    img = torch.stack(img_tensors).squeeze()

    return {
        'txt_g':txt_g,
        'img':img,
        'pdf_hash':pdf_hash
    }

def execute_epochs(dataload, iter):
    checkpoints = '/content/drive/MyDrive/Summer_Project-ECE_697/src_code/checkpoints/model_vit.pt'

    gc.collect()
    losses = []

    model = VisionTransformer()
    #optimizer = torch.optim.SGD(model.parameters(), lr=10**-4)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = 1e-4, 
        weight_decay=10e-6,
        betas=(0.5, 0.999)
        )
    start = 0
    if os.path.exists(checkpoints):
        checkpoint = torch.load(checkpoints)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['iter']
        loss = checkpoint['loss']

    for i in tqdm(range(start, start+iter)):
        df = dataload.get_batches( batch_size = 25, train = True)
        batches = _get_batch_tensors(df)

        img_g= model.encode(batches['img'])
        loss = model.calc_loss(img_g, batches['txt_g'])
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss)

        gc.collect()
        torch.cuda.empty_cache()
        
    with open(f"/content/drive/MyDrive/Summer_Project-ECE_697/src_code/checkpoints/loss_vit/loss_{start}.pickle", "wb") as f:
        pickle.dump(losses, f)

    torch.save({
            'iter': i,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoints)
    
    return f"completed {i}"