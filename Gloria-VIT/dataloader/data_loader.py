"""

This class is used to get the data during train and test 

"""
import pickle, os, pathlib, re, numpy as np, pandas as pd, tqdm, glob, gc

from google.colab import drive
drive.mount('/content/drive/')

class DataLoader:
    def __init__(self, 
                csv_file, 
                img_root_dir, 
                text_root_dir, 
                chexpert_dir):
        
        self.csv_file = csv_file
        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir
        self.chexpert_dir = chexpert_dir
        self.last_read_train_idx = 0
        self.last_read_test_idx = 0

        self.data_df = self.get_data()
        self.train_df = None
        self.test_df = None
        self.train_idx = None
        self.test_df = None
        self.train_idx = None
        self.test_idx = None

    def get_data(self):  
        if not os.path.exists(self.csv_file):
            # get the image embeddings path
            img_files = glob.glob(os.path.join(self.img_root_dir, "*.pickle"))
            img_em = []
            for file in img_files:
                if file.split('/')[-1].startswith('p10'):
                    img_em.append(file)

            img_file_study_ids = [img_em[i].split('/')[-1].split('.')[0].split('_')[-2][1:] for i in range(len(img_em))]

            df1 = pd.DataFrame()
            df1['img_emd'] = img_em
            df1['study_id'] = img_file_study_ids

            # get the text embeddings path
            files = glob.glob(os.path.join(self.text_root_dir, '*.pickle'))
            txt_em = []
            for file in files:
                if file.split('/')[-1].startswith('p10'):
                    txt_em.append(file)
            text_file_study_ids = [txt_em[i].split('/')[-1].split('.')[0].split('_')[-1][1:] for i in range(len(txt_em))]

            df2 = pd.DataFrame()
            df2['txt_emd'] = txt_em
            df2['study_id'] = text_file_study_ids

            # get the chexpert file
            chexpert_df = pd.read_csv(self.chexpert_dir)

            # merge both the dfs
            df1['study_id'] = df1['study_id'].map(str)
            df2['study_id'] = df2['study_id'].map(str)

            df = df1.merge(df2, how='right', left_on = 'study_id', right_on = 'study_id')
            data = df.dropna(how='any').reset_index(drop=True)

            # merge both the dfs and the labels
            data['study_id'] = data['study_id'].map(str)
            self.chexpert_df['study_id'] = self.chexpert_df['study_id'].map(str)

            self.data_df = data.merge(chexpert_df, how='right', left_on = 'study_id', right_on = 'study_id').reset_index(drop=True)
            self.data_df = self.data_df.dropna(subset=['img_emd', 'txt_emd']).reset_index(drop=True)
            print('creating data csv file...')
            self.data_df.to_csv(self.csv_file)

        else:
            print('Fetching data from csv...')
            self.data_df = pd.read_csv(self.csv_file)
        
        return self.data_df

    def train_test_split(self, test_ratio = 0.2): # randomize and get train and test data
        #self.data_df = self.get_data()
        n = len(self.data_df)
        train_idx = np.ceil((1-test_ratio)*n)
        test_idx = train_idx+1

        self.train_df = self.data_df.loc[:train_idx]
        self.test_df = self.data_df.loc[test_idx:]

        # randomize the index of train dataframe
        train_idx_shuffle = np.arange(len(self.train_df))
        self.train_idx = np.random.RandomState(seed=1).permutation(train_idx_shuffle) 

        # randomize the index of test dataframe
        test_idx_shuffle = np.arange(max(self.train_df.index)+1, max(self.test_df.index)+1)
        self.test_idx = np.random.RandomState(seed=1).permutation(test_idx_shuffle)
        print('Spliting to train and test data...')

    # get the train and test items in get_item()
    
    def get_batches(self, batch_size = 25, train = True):
        batch_idxs = self._get_batch_idxs(batch_size , train)

        batch_df = pd.DataFrame()
        for idx in batch_idxs:
            batch_df = batch_df.append(self.__getitem__(idx))
        #print(f"batch of {len(batch_df)} returned !")
        return batch_df

    def __getitem__(self, idx):
        return self.data_df.loc[int(idx)]

    def _get_batch_idxs(self, batch_size, train):
        
        batch_idx = []
        if train:
            if len(self.train_idx) - self.last_read_train_idx < batch_size:
                batch_idx.extend(self.train_idx[self.last_read_train_idx:])
                if len(batch_idx) < batch_size:
                    n = batch_size - len(batch_idx)
                    batch_idx.extend(self.train_idx[:n])
                    self.last_read_train_idx = n+1
            else:
                n = self.last_read_train_idx + batch_size
                batch_idx.extend(self.train_idx[self.last_read_train_idx:n])
                self.last_read_train_idx = n
                
        else:
            if len(self.test_idx) - self.last_read_test_idx < batch_size:
                batch_idx.extend(self.test_idx[self.last_read_test_idx:])
                if len(batch_idx) < batch_size:
                    n = batch_size - len(batch_idx)
                    batch_idx.extend(self.test_idx[:n])
                    self.last_read_test_idx = n+1
            else:
                n = self.last_read_test_idx + batch_size
                batch_idx.extend(self.test_idx[self.last_read_test_idx:n])
                self.last_read_test_idx = n

        return batch_idx

