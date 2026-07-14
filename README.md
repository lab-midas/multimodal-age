# Multimodal contrastive regression for organ-resolved biological age prediction
This repository contains the implementation of the MICCAI 2026 paper #3491 "Multimodal contrastive regression for organ-resolved biological age prediction" by Veronika Ecker, Tim Förster, Sergios Gatidis, Thomas Küstner, and Bin Yang.

We propose a multimodal regression framework for estimating biological age. Estimation is performed using MRI images, as well as tabular data from the UK Biobank. Using multimodal data strengthens the robustness and interpretability of biological age prediction.

## Preparation
First, install the required packages detailed in the [requirements.txt](requirements.txt) file. The preprocessing files of the imaging data can be found in [this repository](https://github.com/lab-midas/biological_age).

## Training
For training, run [main.py](main.py) using the following flags:

| FLAG | True | False | Values |
| ------------- | ------------- | ------------- |------------- |
| train  | train a model  | | |
| only_finetuning  | finetune a model  | | |
| gradients  | calculate integrated gradients of a model  | | |
| compare_embeddings  | create plots of the loss space from two different models for comparision  | | |
| hyperparameter_tuning  | use hyperparameter tuning  | run a single training with predefined model configs | |
| use_wandb | use wandb to display the training progress (need to be True for hyperparameter_tuning) | log training process only to file | |
| direct_evaluation | evaluate model after training | train without evaluation afterwards | |
| data_type |  |  | "multimodal", "tabular", "image" | 
| organ |  |  | "brain", "heart", "liver", "pancreas", "spleen", "kidney" | 

Note: only one of the following flags should be set: "train", "only_finetuning", "gradients" or "compare_embeddings". If all four flags are false, only an evaluation of a loaded model will be performed.

Example for a multimodal brain training run with hyperparameter tuning:
```
python main.py --train=True --hyperparameter_tuning=True --data_type="multimodal" --organ="brain" --direct_evaluation="True".
```