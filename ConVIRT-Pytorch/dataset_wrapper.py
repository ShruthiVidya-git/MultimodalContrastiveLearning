# Mount Drive
from google.colab import drive
drive.mount('/content/drive/')
import sys
sys.path.insert(0,'/content/drive/MyDrive/Summer_Project-ECE_697/ConVIRT Pytorch')
from dataset import ClrDataset

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
#from dataloader.gaussian_blur import GaussianBlur
from torchvision import datasets
#from .dataset import ClrDataset

np.random.seed(0)


class DataSetWrapper(object):
    def __init__(self, 
                batch_size, 
                num_workers, 
                valid_size, 
                input_shape, 
                s, 
                csv_file, 
                img_root_dir, 
                img_path_col, 
                text_col, 
                text_from_files, 
                text_root_dir):
                
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.csv_file = csv_file
        self.img_root_dir = img_root_dir
        self.img_path_col = img_path_col 
        self.text_col = text_col
        self.text_from_files = text_from_files
        self.text_root_dir = text_root_dir

        
    def get_data_loaders(self):
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = ClrDataset(csv_file=self.csv_file,
                                    img_root_dir=self.img_root_dir,
                                    input_shape = self.input_shape,
                                    img_path_col = self.img_path_col, 
                                    text_col = self.text_col, 
                                    text_from_files = self.text_from_files, 
                                    text_root_dir = self.text_root_dir, 
                                    transform=SimCLRDataTransform(data_augment)
                                    )

        # train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
        #                                transform=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([
                                              transforms.Resize((self.input_shape[0], self.input_shape[1])),
                                              transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                            #   transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                            #   GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform_image):
        self.transform_image = transform_image

    def __call__(self, sample):
        xi = self.transform_image(sample['image'])
        xl = sample['phrase']

        return xi, xl
