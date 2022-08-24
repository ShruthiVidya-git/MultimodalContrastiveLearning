# MultimodalContrastiveLearning
This repository contains the implementation of multimodal contrastive learning models for disease classification
## Dataset
Our official data was provided us to by Prof. Aan Mcmillan
The dataset is available at https://physionet.org/content/mimic-cxr/2.0.0/
## Problem Statement
One of the biggest challenges in the field of Medical AI is the indispensable requirement to have huge labeled data which is challenging due to its sensitive nature. Medical data is very difficult to collect and annotate which is addressed in this work by using a self-supervised learning algorithm which requires no labeling or annotations for training. Integration of multi-modal data, which is inherent in the nature of healthcare records, is the next reasonable step in using Deep Learning for Biomedical diagnosis.
During training, image and text encoders are used to convert the radiographs to image embeddings,textual reports to word embeddings and get their latent representations. Contrastive learning techniques can then be applied to learn the joint embedding space and zero shot classification can be
implemented at the inference to identify diseases.
