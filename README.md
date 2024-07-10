# Master's thesis on Text classification

### Participation in challenge PAN 2024 "Oppositional thinking analysis: Conspiracy theories vs critical thinking narratives"

  

## Introduction

This master's thesis compares various Neural Network models in the context of a specific Natural Language Processing task: classifying texts as either conspiracy or rational thinking. Conducted with the support of the AI+DA research group at the Universidad Politécnica de Madrid, and developed thanks to the public open dataset provided by the PAN24 challenge.

The master thesis focuses on trying to find a model with the best possible performance in the classification task on the dataset given by the competition. In this search, several tests are performed comparing models based on the transformer architecture, various loss functions and data augmentation techniques.

In this repository you can find all the code used for the experiments. Below you can find a brief description of each directory and code file.

  
  

## Data

The /dataset_oppositional directory contains the data from which the experiments will be performed. First we find the files "dataset_train_en" and "dataset_train_es" that refer to the original datasets of the competition in their English and Spanish versions respectively.

Then we find the files "train_es_data", "train_en_data", "test_es_data" and "test_en_data". These files are a product of processing the original data. First of all, only the text and classification information is retained, eliminating the text segmentation information related to other tasks of the competition. Finally the data sets are split in a 90/10 ratio respecting the same class imbalance in both files.

Next file "train_en_data_AUG_1-2.pth" is the result of experimentation in data augmentation, where we tried to increase the diversity of the data by rephasing with LLama-3 8B. In the same way "train_en_translated_data.pth" is the result of the translation of the Spanish dataset into English made by Llama-3 8B.

Finally the /submission directory only contains a json file wich was the results we sent to the competition

  

## Code Files Explained

  

#### CreateDataset.py
In this file there are four torch.utils.data.dataset functions that transform the data for various purposes

  

- **createAuxDataset**

	This function just parse the original data and take the text and category

- **createAuxSubmissionDataset**

	This function is similar to the previous one but applied to the competition evaluation data. It differs in that this function does not extract the category since it does not come, since it is what should be predicted. It also incorporates directly the tokenization of Bert since in the case of the final submission we already knew that the model chosen was Bert.

- **BertDataset**

This was designed to make the experiments with Bert-like models. It takes as input the text and the label, tokenize the text and apply diferent data augmentations if is call to do so.

  

- **RobertaDataset**

Just the same as before but without data augmentation, and with roberta tokenizer

  

### llama3.ipynb

llama3 file includes the experiments done with LLMs, that only includes an attemp of data rephrasing and the spanish dataset translation.

  

### models.py

In this file you will find the declaration of all the models that have been tested. You will find Bert-base, Bert-large, Roberta models. When reference is made to covid in the name, it means that it is not based on the standard hugging face weights, but is a pre-trained version of covid texts.

It is fair to say that these models are sometimes repeated with a normal version and another one with the addition of mixup in the name. This is due to the fact that in the initial experiments with this technique the models were implemented with a normal forward, and an added "mixup_forward" method.

However, when scaling the size of the models and the batch we found that it was not possible to parallelize in pytorch, or at least not in a simple way, when you wanted to use a forward method other than the one called "forward".

  

### tfm-conspiracy.ipynb

This is the notebook where I conducted all the experiments.

All the configuration can be done by changing the variables in the configuration notebook cell
Its configured to train with 10 different seeds a 5-fold-validation model ensemble. It save each locally, and evaluate each model and 5-fold ensemble.

Below we can find code to load checkpints and evaluate ensembles.

Finally we can find code related with unknown threshold MCC classification exploration, and code to investigate if filtering models of the ensemble with low valiadation MCC had impact on the ensemble (did not).

  
  

###  utils.py

We can find here a mix of unrelated code. I know, but is what it is.

First we can find the **train_loop** and **evaluate methods**, both consist of a lot of possible experimental configurations in terms of hyperparameters and loss_functions

Next we find **evaluate_kfold_ensemble** that is a function that calculates the MCC of the mean of the predictions of the ensemble. (Bagging)

  
- **get_mixup_batch** 
	Then we find **get_mixup_batch** wich sample for each element of the batch a random datapoint of the other class.

- **mix_labels**
	After this the function **mix_labels** mix the labels with the beta distribution that have been set.

  

- **contrastive_loss**, **torch_contrastive_loss** and **new_torch_contrastive_loss**.

	**contrastive_loss** is a code got from another open repository, that try to replicate the paper "SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINED LANGUAGE MODEL FINE-TUNING". However, **the implementation is wrong**. I keep it since I did use it to experimet after fix it.
**torch_contrastive_loss** is the corrected function. As far as I understand the previous one was not implemented in torch so it was not traking the gradients from the contrastive term of the loss fucntion.
	**new_torch_contrastive_loss** this is the same as before, but I added a division of the contrastive term by the batch size, since in the way the paper implements it, the batch size do affect a lot the size of the loss and the gradients, wich difficult to compare experiments with different batch sizes.