"""
Code Adopted from Gloria Repo https://github.com/marshuang80/gloria

This file consists of the text preprocessing and encoding steps to generate and store the text embeddings from reports
"""

!pip install transformers

# Add required libraries
import os, csv, pandas as pd, numpy as np, tqdm, torch, re, nltk, pickle
from nltk.tokenize import RegexpTokenizer as word_tokenizer
from transformers import AutoTokenizer, BertModel, AutoModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

# mount drive to store files
from google.colab import drive
drive.mount('/content/drive/')

# download stop words, lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bertmodel = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)

        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.batch_size = 4


    def word_sent_embeddings(self):
        return self.ixtoword

    def bert_write(self, ids, attn_mask, token_type, embedding_path):
        outputs = []
        start = 0
        end = 41
        idx = 0
        while end <= len(ids):
            output = self.bertmodel(ids[start:end], attn_mask[start:end], token_type[start:end])
            start += 41
            end += 41
            picklefile = embedding_path + str(idx) + '.pickle'
            idx += 1 
            with open(picklefile, 'wb') as pickle_file:
                pickle.dump(output, pickle_file)
            #outputs.append(output)

    def text_tokenization(self,tokens_hash_path, pdf_hash=[], sents=[]):
        if not os.path.exists(tokens_hash_path):
            print('Creating caption tokens...')
            tokens = self._get_text_tokens(sents, pdf_hash)
            # store the text file paths in pickle 
            with open(tokens_hash_path, 'wb') as pickle_file:
                pickle.dump(tokens, pickle_file)
            # pickle_file = open(tokens_hash_path,"wb")
            # pickle.dump(tokens, tokens_hash_path)
            # pickle_file.close()

        else:
            print('loading caption tokens from pickle file...')
            # load pickle file
            pickle_file = open(tokens_hash_path, "rb")
            tokens = pickle.load(pickle_file)
            pickle_file.close()

        return tokens

        
    """"tokenize and get embeddings"""
    def _get_text_tokens(self, sents, pdf_hash):
        processed_text_tensors = []
        for idx, tokens in tqdm.tqdm(enumerate(sents)):
            sent = []
            sent.append(" ".join(tokens))
            text_tensors = self.tokenizer(sent,
                            return_tensors="pt", # return pytorch tensors
                            truncation=True,
                            padding="max_length",
                            max_length=512
                        )
            # get the actual word corresponding to the tokenized text
            text_tensors["sent"] = [self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()]
            
            text_tensors["pdf_hash"] = pdf_hash[idx]
            processed_text_tensors.append(text_tensors)
        
        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack([x["attention_mask"] for x in processed_text_tensors])
        token_type_ids = torch.stack([x["token_type_ids"] for x in processed_text_tensors])
    
        caption_ids = caption_ids.squeeze()
        attention_mask = attention_mask.squeeze()
        token_type_ids = token_type_ids.squeeze()

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "pdf_hash":pdf_hash
        }
    
    """
    Preprocess and get the tokens for all images

    caption_mapping -> {'pdf_hash' : 'impressions extract'}
    """
    def _text_preprocessing_word_tokenizer(self, caption_mapping, captions, pdf_hash, pickle_path):    
        # Split sentences into words with maximum 
        max_word_count = 97
        sentences, words, len_sentence, num_sentences, to_remove  = [], [], [], [], []

        # get all the stop words
        stop_words = set(stopwords.words('english'))
        word_lemmatizer = WordNetLemmatizer()

        img_hash_to_sentence = {}

        for idx, row in tqdm.tqdm(captions.iterrows(), total=captions.shape[0]): 
            
            if not row.isnull()[0] :  
                # split by numbers folowed by special characters
                
                split_seq = re.compile("[0-9]+\.")
                caption = split_seq.split(row[0])              
                # split by "." to get seperate sentences
                caption = [sentence.split(".") for sentence in caption]
                # combine all sentences as a single list per caption
                caption = [sent for sentence in caption for sent in sentence]

                for row_caption in caption:
                    # ignore empty captions 
                    if len(row_caption) == 0:
                        continue

                    # replace if there is no representation in unicode
                    row_caption = row_caption.replace("\ufffd\ufffd", " ")
                    
                    # convert everything to lower case
                    row_caption = row_caption.lower()

                    # get all alphanumeric characters as tokens - separate words
                    tokenizer = word_tokenizer(r"\w+")
                    word_tokens = tokenizer.tokenize(row_caption)

                    # ignore if the sentence has less than 3 words 
                    if len(word_tokens) < 3 :
                        continue

                    sent_filtered = []
                    count_words = 0
                    for word in word_tokens:
                        # Lemmatize all words to root 
                        word = word_lemmatizer.lemmatize(word)

                        # remove the stop words
                        if word in stop_words:
                            continue
                        
                        # remove characters and any tokens which aren't english
                        included_tokens = []
                        word = word.encode("ascii", "ignore").decode("ascii")
                        if len(word) > 0:
                            included_tokens.append(word)
                        
                        count_words += len(included_tokens)
                        if count_words >= max_word_count:
                            break
                        sent_filtered.append(" ".join(included_tokens))    
                    len_sentence.append(len(included_tokens))
                num_sentences.append(len(sent_filtered))
                if len(sent_filtered) < max_word_count:
                    img_hash_to_sentence[pdf_hash[idx]] = sent_filtered
                else:
                    to_remove.append(pdf_hash[idx])
            else:
                to_remove.append(pdf_hash[idx])
        print('\n\n',len(img_hash_to_sentence),'captions preprocessed')

        # store the text file paths in pickle file
        pickle_file = open(pickle_path,"wb")
        pickle.dump([img_hash_to_sentence, to_remove], pickle_file)
        pickle_file.close()

        return img_hash_to_sentence, to_remove

    def get_text_tokenizer(self):
        pass

    def get_text_reports(self, reports_path, pickle_path):
        if not os.path.exists(pickle_path):
            print('Fetching all the text reports...')
            txt_files = []
            for root, dirs, files in os.walk(reports_path):
                for file in files:
                    if file.endswith(".txt"):
                        path_file = str(root)+'/'+str(file)
                        txt_files.append(path_file)
            # store the text file paths in pickle file
            pickle_file = open(pickle_path,"wb")
            pickle.dump(txt_files, pickle_file)
            pickle_file.close()

        else:
            print('loading reports from pickle file...')
            # load pickle file
            pickle_file = open(pickle_path, "rb")
            txt_files = pickle.load(pickle_file)
            pickle_file.close()

        return txt_files

    def _get_captions(self, txt_files):
        captions = {}
        for file in tqdm.tqdm(txt_files): 
            impressions_extract = []
            with open(file) as f:
                contents = f.read()
                if len(contents) >0 :
                    start_idx = re.search(r'IMPRESSION', contents)
                    if start_idx:
                        if start_idx.end() > 1:
                            impressions_extract.append(contents[start_idx.end()+1 : len(contents)])
            captions[file.split('files/')[1]] = impressions_extract
        return captions

    def get_impressions(self, txt_files, captions_file):
        if not os.path.exists(captions_file):
            print('Fetching impressions from reports...')
            captions = self._get_captions(txt_files)
        else:
            print('loading impressions from pickle file...')
            # load pickle file
            pickle_file = open(captions_file, "rb")
            captions = pickle.load(pickle_file)
            pickle_file.close()
        return captions

if __name__ == "__main__":
    text_encoder_obj = TextEncoder()
    reports_path = '/content/drive/MyDrive/mimic_text_data'
    txt_files_path = '/content/drive/MyDrive/Summer_Project-ECE_697/all_file_names.pickle'
    captions_file = '/content/drive/MyDrive/captions.pickle'
    processed_file_path = '/content/drive/MyDrive/impressions_processed.pickle'
    tokens_path = '/content/drive/MyDrive/tokens.pickle'
    tokens_hash_path = '/content/drive/MyDrive/tokens_hash_512.pickle'
    embedding_ops = '/content/drive/MyDrive/Summer_Project-ECE_697/Embedding_outputs/'

    if not os.path.exists(tokens_hash_path):
        print('creating the processed.pickle file')
        txt_files = text_encoder_obj.get_text_reports(reports_path, txt_files_path)
        captions = text_encoder_obj.get_impressions(txt_files, captions_file)
        img_hash_to_sentence, to_remove = text_encoder_obj._text_preprocessing_word_tokenizer( captions, 
                                pd.DataFrame(captions.values()), list(captions.keys()), processed_file_path)
        pickle_file = open(processed_file_path, "rb")
        [img_hash_to_sentence, to_remove] = pickle.load(pickle_file) 
        pickle_file.close() 

        sents = list(img_hash_to_sentence.values())
        tokens = text_encoder_obj.text_tokenization(tokens_hash_path, list(img_hash_to_sentence.keys()),sents) 
    else:
        print('loading from pickle file...')
        tokens = text_encoder_obj.text_tokenization(tokens_hash_path)

        ids = tokens['caption_ids']
        attn_mask = tokens['attention_mask']
        token_type = tokens['token_type_ids']
        pdf_hash = tokens['pdf_hash']
        
        r = re.compile("p10/")
        paths = list(filter(r.match, pdf_hash))
    idxtoword = text_encoder_obj.word_sent_embeddings()

bertmodel = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)
embedding_path = '/content/drive/MyDrive/Summer_Project-ECE_697/Embedding_ops_temp/'

def get_local_global_text_embeddings(pdf_hash):
    embedding_ops = '/content/drive/MyDrive/Summer_Project-ECE_697/Embedding_outputs/'
    text_emds_path = '/content/drive/MyDrive/Summer_Project-ECE_697/'

    r = re.compile("p10/")
    paths = list(filter(r.match, pdf_hash))

    p10_file_counts = len(paths)//41 + 1
    print('start')
    # getting the corresponding pdf hash

    local_embed = []
    global_embed = []
    sentences_list = []
    start = 0
    end = start + 41
    for i in tqdm.tqdm(range(p10_file_counts)): # 448
        
        # get the bert encoding with outputs from all layers from pickle
        bert_embed_path_ = embedding_ops + str(i) + '.pickle'

        with open(bert_embed_path_, 'rb') as pickle_file:
            output = pickle.load(pickle_file)

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

        """ starts here """    

        # loop over batch
        batch_size, num_layers, num_words, dim = embeddings.shape

        embeddings = embeddings.permute(0, 2, 1, 3)

        agg_embs_batch = []
        sentences = []
            

        for embs, caption_id in zip(embeddings, caption_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence

            """Add start stop tokens for padding"""
            for word_emb, word_id in zip(embs, caption_id):

                word = idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        # batch size first
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        # layers_first reshape
        embeddings = agg_embs_batch.permute(1, 0, 2, 3)

        # sentence level embeddings 
        sent_embeddings = embeddings.mean(axis=2)
        sent_embeddings = sent_embeddings.sum(axis=0)
        #print(sent_embeddings.shape)

        # sum as aggregate method to get the word and sentence embeddings

        word_embeddings = embeddings.sum(axis=0)
        #print(word_embeddings.shape)
        
        # store each of the global, local, sentences and the respective text file name in pickle
        local_embed.append(word_embeddings)
        global_embed.append(sent_embeddings)
        sentences_list.append(sentences)
        text_embed = text_emds_path + str(start) + '.pickle'

        # start = end
        # end += 41
        print('running')
        with open(text_embed, 'wb') as pickle_file:
            pickle.dump({'text_local_embeddings':local_embed, 'text_global_embeddings':global_embed, 
                'sentences':sentences_list}, pickle_file)
            
        for idx in range(batch_size): # 41
            local_embed.append(word_embeddings[idx])
            global_embed.append(sent_embeddings[idx])
            sentences_list.append(sentences[idx])

            # local_embed = [word_embeddings[idx]]
            # global_embed = [sent_embeddings[idx]]
            # sentences_list = [sentences[idx]]
        text_embed = text_emds_path + str(start) + '.pickle'
        with open(text_embed, 'wb') as pickle_file:
            pickle.dump({'text_local_embeddings':local_embed, 'text_global_embeddings':global_embed, 
                    'sentences':sentences_list, 'pdf_hash':pdf_hash[start:end]}, pickle_file)
        
        #update the start and end index for pdf_hash
        start = end
        end += 41

