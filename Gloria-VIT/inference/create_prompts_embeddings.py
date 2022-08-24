"""
Adopted from gloria repository

This file enlists the steps to create the class prompts to find the closest class at inference to the test image
"""

!pip install transformers

from google.colab import drive
drive.mount('/content/drive')

import torch, torch.nn as nn, cv2, numpy as np, pickle, os, re, tqdm, random, sys, pandas as pd
from sklearn import metrics
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import RegexpTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.insert(0,'/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Models/')
import gloria_model
from gloria_model import Gloria

sys.path.insert(0,'/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Datasets/')
import data_loader
from data_loader import DataLoader

sys.path.insert(0,'/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Inference/')
import similarities



def generate_chexpert_class_prompts(n: int = 5):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        prompts[k] = random.sample(cls_prompts, n)
    return prompts

def process_class_prompts(class_prompts, device):
    cls_2_processed_txt = {}
    for k, v in class_prompts.items():
        cls_2_processed_txt[k] = process_text(v, device)

    return cls_2_processed_txt

def process_text(sents, device):
    processed_text_tensors = []
    cap_lens = []
    for idx, tokens in tqdm.tqdm(enumerate(sents)): # each sentence in all classes
        sent_cap = []
        sent = []
        sent.append("".join(tokens))
        
        text_tensors = tokenizer(sent,
                        return_tensors="pt",
                        truncation=True,
                        padding="max_length",
                        max_length=512
                    )

        text_tensors["sent"] = [ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()]
        processed_text_tensors.append(text_tensors)
        
        sent_cap.append([txt for txt in text_tensors["sent"] if not txt.startswith("[") ] )
        cap_lens.append(len(sent_cap[0]))
    
    caption_ids = torch.stack([x["input_ids"].squeeze() for x in processed_text_tensors])
    attention_mask = torch.stack([x["attention_mask"].squeeze() for x in processed_text_tensors])
    token_type_ids = torch.stack([x["token_type_ids"].squeeze() for x in processed_text_tensors])
    
    return {
        "caption_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "cap_lens": cap_lens,
    }

def get_local_global_prompts(outputs):  
    local_embed = {}
    global_embed = {}

    for i in tqdm.tqdm(list(outputs.keys())):
        output = outputs[i]
        # get the last 4 layer ouputs to get a sensible context for the report
        last_n_layers = 4
        embedding_dim = 768
        # aggregate intermetidate layers

        all_embeddings = output[2]
        embeddings = torch.stack(
            all_embeddings[-last_n_layers :]
        )  # layers, batch, sent_len, embedding size

        embeddings = embeddings.permute(1, 0, 2, 3)
        
        # loop for each of the report and aggregate the local and global features
        # loop over batch
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        
        # sentence level embeddings 
        word_embeddings = embeddings.mean(axis=2)
        # print(word_embeddings.shape)
        word_embeddings = word_embeddings.sum(axis=0)
        # sum as aggregate method to get the word and sentence embeddings
        sent_embeddings = word_embeddings.sum(axis=0)
        # print(sent_embeddings.shape)
        # store each of the global, local, sentences and the respective text file name in pickle
        local_embed[i] = word_embeddings
        global_embed[i] = sent_embeddings

    return local_embed, global_embed

def create_prompts_embeddings():   
    prop = generate_chexpert_class_prompts()
    processed_prompts = process_class_prompts(prop, device)
    classes = list(processed_prompts.keys())
    outputs = {}
    cap_lens_avg = {}

    for cls in classes:
        output = bertmodel(processed_prompts[cls]['caption_ids'], processed_prompts[cls]['attention_mask'], processed_prompts[cls]['token_type_ids'])
        outputs[cls] = output
        cap_lens_avg[cls] = np.ceil(np.mean(processed_prompts[cls]['cap_lens']))

    with open("/content/drive/MyDrive/Summer_Project-ECE_697/src_code/checkpoints/bert_embeddings_captions1.pickle", "wb") as f:
        pickle.dump(outputs, f)

    local_embed, global_embed = get_local_global_prompts(outputs)
    with open("/content/drive/MyDrive/Summer_Project-ECE_697/src_code/checkpoints/captions_local_global_6_classes.pickle", "wb") as f:
        pickle.dump({'local_embed':local_embed, 'global_embed':global_embed, 'cap_lens_avg':cap_lens_avg}, f)
    print('\ncreating caption embeddings..')
    return {'local_embed':local_embed, 'global_embed':global_embed, 'cap_lens_avg':cap_lens_avg}

def get_embeddings():
    embed_captions = "/content/drive/MyDrive/Summer_Project-ECE_697/src_code/checkpoints/captions_local_global_6_classes.pickle"

    if not os.path.exists(embed_captions):
        embed = create_prompts_embeddings()
    else:
        print('\nfetching prompt embeddings...')
        with open(embed_captions, "rb") as f:
            embed = pickle.load(f)
    return embed

def _get_batch_tensors(df):
    """
    Get batches of image and text tensors
    """
    img_paths = list(df['img_emd'])
    
    img_tensors = []
    paths = []
    for idx in range(len(df)):
        # read batch of 25 image tensors
        with open(img_paths[idx], 'rb') as f:
            img_tensor = pickle.load(f)
        tensor_out = tensor_reshape(img_tensor)

        img_tensors.append(tensor_out)
        paths.append(img_paths[idx].split('/tensor_image/')[1].split('.pickle')[0].replace('_', '/'))
    img = torch.stack(img_tensors).squeeze()

    return img, paths

def tensor_reshape(img_tensor):

    if img_tensor.shape[2] != 256:
        tensor_out = nn.functional.pad(input=img_tensor, pad=(0, 0, 1,0) , mode='constant', value=0)
        return tensor_out

    elif img_tensor.shape[3] != 256:
        tensor_out = nn.functional.pad(input=img_tensor, pad=(0, 1, 0,0) , mode='constant', value=0)
        return tensor_out

    return img_tensor

if __name__=="__main__":   
    # get the files for getting the test data
    #test_path = '/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Inference/test.csv'
    test_path = '/content/drive/MyDrive/Summer_Project-ECE_697/src_code/Inference/test_true.csv'

    # get the recent copy of the model checkpoint
    checkpoints = '/content/drive/MyDrive/Summer_Project-ECE_697/src_code/checkpoints/model.pt'

    # Define class prompts based on severity, subtype and location
    CHEXPERT_CLASS_PROMPTS = {
        "Pneumonia": {
            "severity": [
                "mild",
                "moderate",
                "severe",
            ],
            "subtype": [
                "multifocal pneumonia",
                "pcp pneumonia",
                "viral pneumonia",
                "concurrent pneumonia",
                "superimposed pneumonia",
                "recurrent basal pneumonia",
                "underlying pneumonia",
                "right basilar pneumonia",
                "Pneumoperitoneum",
                "acute pneumonia",
                "aspiration pneumonia",
                "bacterial pneumonia",
                "mycloplasma pneumonia",
                "fungal pneumonia",
            ],
            "location": [
                "in the right lower lobe",
                "in the middle lower lobe",
                "in the left lower lobe",
                "in the right middle lobe",
            ],
        },

        "Atelectasis": {
            "severity": ["", "mild", "minimal"],
            "subtype": [
                "subsegmental atelectasis",
                "linear atelectasis",
                "trace atelectasis",
                "bibasilar atelectasis",
                "retrocardiac atelectasis",
                "bandlike atelectasis",
                "residual atelectasis",
            ],
            "location": [
                "at the mid lung zone",
                "at the upper lung zone",
                "at the right lung zone",
                "at the left lung zone",
                "at the lung bases",
                "at the right lung base",
                "at the left lung base",
                "at the bilateral lung bases",
                "at the left lower lobe",
                "at the right lower lobe",
            ],
        },
        "Cardiomegaly": {
            "severity": [""],
            "subtype": [
                "cardiac silhouette size is upper limits of normal",
                "cardiomegaly which is unchanged",
                "mildly prominent cardiac silhouette",
                "portable view of the chest demonstrates stable cardiomegaly",
                "portable view of the chest demonstrates mild cardiomegaly",
                "persistent severe cardiomegaly",
                "heart size is borderline enlarged",
                "cardiomegaly unchanged",
                "heart size is at the upper limits of normal",
                "redemonstration of cardiomegaly",
                "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
                "cardiac silhouette size is mildly enlarged",
                "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
                "heart size remains at mildly enlarged",
                "persistent cardiomegaly with prominent upper lobe vessels",
            ],
            "location": [""],
        },
        "Consolidation": {
            "severity": ["", "increased", "improved", "apperance of"],
            "subtype": [
                "bilateral consolidation",
                "reticular consolidation",
                "retrocardiac consolidation",
                "patchy consolidation",
                "airspace consolidation",
                "partial consolidation",
            ],
            "location": [
                "at the lower lung zone",
                "at the upper lung zone",
                "at the left lower lobe",
                "at the right lower lobe",
                "at the left upper lobe",
                "at the right uppper lobe",
                "at the right lung base",
                "at the left lung base",
            ],
        },
        "Edema": {
            "severity": [
                "",
                "mild",
                "improvement in",
                "presistent",
                "moderate",
                "decreased",
            ],
            "subtype": [
                "pulmonary edema",
                "trace interstitial edema",
                "pulmonary interstitial edema",
            ],
            "location": [""],
        },
        "Pleural Effusion": {
            "severity": ["", "small", "stable", "large", "decreased", "increased"],
            "location": ["left", "right", "tiny"],
            "subtype": [
                "bilateral pleural effusion",
                "subpulmonic pleural effusion",
                "bilateral pleural effusion",
            ],
        },
    }
    # get the prompt embeddings
    embed = get_embeddings()