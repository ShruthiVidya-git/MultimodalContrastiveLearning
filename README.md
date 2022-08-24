# MultimodalContrastiveLearning
This repository contains the implementation of multimodal contrastive learning models for disease classification
## Dataset
Our official data was provided us to by Prof. Aan Mcmillan
The dataset is available at https://physionet.org/content/mimic-cxr/2.0.0/
## Problem Statement
One of the biggest challenges in the field of Medical AI is the indispensable requirement to have huge labeled data which is challenging due to its sensitive nature. Medical data is very difficult to collect and annotate which is addressed in this work by using a self-supervised learning algorithm which requires no labeling or annotations for training. Integration of multi-modal data, which is inherent in the nature of healthcare records, is the next reasonable step in using Deep Learning for Biomedical diagnosis.
During training, image and text encoders are used to convert the radiographs to image embeddings,textual reports to word embeddings and get their latent representations. Contrastive learning techniques can then be applied to learn the joint embedding space and zero shot classification can be
implemented at the inference to identify diseases.

## Architecture Pipeline 

![Alt text](https://github.com/ShruthiVidya-git/MultimodalContrastiveLearning/blob/main/Results/Flowchart.jpeg "Flow chart of our workflow")

## Code Structure 

> ```Gloria-VIT``` <br />
>  > ```Models```<br />
>  > >```Gloria``` <br />
>  > > >```gloria_model.py``` Models defined with ResNet50 and BioClinicalBERT as Image and Text Encoder <br />
>  > > >```train.py``` Fine-tunes the ResNet50 and BioClinicalBERT for MIMIC-III Image-Text pairs<br />  
>  > > >
>  > >```Vit```<br />
>  > > >```train_vit.py``` Models defined with ViT and BioClinicalBERT as Image and Text Encoder <br />
>  > > >```vit.py``` Fine-tunes the ViT and BioClinicalBERT for MIMIC-III Image-Text pairs<br />
>  > > >
>  > ```dataloader```<br />
>  > >```data_loader.py``` Data Loader which fetches data for training and testing the models<br />
>  > >
>  > ```inference```<br />
>  > >```create_prompts_embeddings.py``` This file creates the hand-crafted class prompts <br />
>  > >```similarities.py``` Used to compute the global and local similarities between the test images and class prompts<br />
>  > >
>  > ```utils```<br />
>  > >```text_processing.py``` This contains the text preprocessing steps involved to get the text embeddings<br />
>  > >
>  ```main.ipynb``` Run this to train Gloria Model<br />
>  ```main_vit.ipynb``` Run this to train ViT+BERT Model<br />
>  ```Inference_time.ipynb``` Inference results for both models for Binary and Multi-class Classification<br />
>  ```ConVIRT-Pytorch``` <br />
>  >```dataset.py``` Reads csv and gets the images and phrases<br />
>  >```dataset_wrapper.py``` Performs image tranformation/ augmentation and loads train/ validation data<br />
>  >```model.py``` Definition of Image and Text Encoders<br />
>  >```nt_xent.py```Definition of Loss function<br />
>  >```train.py``` Loads and checkpoints for training the models<br />
>  >```run.ipynb```Runs the model and trains it<br />




## Conclusion
Supervised learning has served as a gateway to all the machine learning enthusiasts for the last
decade. But these traditional supervised learning models are not sufficient in this age of data. With
the growing need of accurate and self sufficient AI models, having supervised learning as the only
training approach is very limiting. In our project, we have experimented with a newer approach of
multimodal self-supervised learning. We through our analysis can conclude that having more than
one modality as the input does improve the model performance. We can also conclude that having
both local and global features improve the efficiency of any given model. Contrastive and Zero shot
learning although difficult in implementation do have many perks to it, the major one being not
needing labeled data and is extremely valuable in the field of medicine.


