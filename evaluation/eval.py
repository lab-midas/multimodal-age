"""File for evaulation"""
import matplotlib.pyplot as plt
import gin
import numpy as np
from utils.utils_disp import display, display_compare, gen_lists, plot_results
import logging
import torch
import torch.nn as nn
import pandas as pd
import os

from torch.amp import autocast


@gin.configurable
def evaluate(model, ds_test, device, organ, two_label, save_paths = None):
    """
    This method evaluates the model by calculating the losses and storing the predictions.
    """
    loss_object = nn.MSELoss().to(device)
    loss_object_MAE = nn.L1Loss().to(device)
    model.eval()
    running_loss_val = 0.0
    running_loss_mae = 0.0
    running_loss_t = 0.0
    running_loss_i = 0.0
    running_loss_t_mae = 0.0
    running_loss_i_mae = 0.0
    results = []
    for batch in ds_test:
        with autocast(device_type=device.type):
            if two_label == True:
                y_i = model(batch, modality="image")
                labels_i = batch['label'].view(-1, 1).to(device)
                loss_i = loss_object(y_i, labels_i)
                loss_mae_i = loss_object_MAE(y_i, labels_i)

                labels_t = batch['label-t'].view(-1, 1).to(device)
                y_t = model(batch, modality="tabular")
                loss_t = loss_object(y_t, labels_t)
                loss_mae_t = loss_object_MAE(y_t, labels_t)

                loss = (loss_i + loss_t) / 2
                loss_mae = (loss_mae_i + loss_mae_t) / 2

                logging.info(f"val loss: {loss}")
                logging.info(f"val loss MAE: {loss_mae}")

                running_loss_t += loss_t
                running_loss_i += loss_i
                running_loss_t_mae += loss_mae_t
                running_loss_i_mae += loss_mae_i
            else:
                labels = batch['label'].view(-1, 1).to(device)
                y = model(batch)
                loss = loss_object(y, labels)
                loss_mae = loss_object_MAE(y, labels)
                logging.info(f"val loss: {loss}")
                logging.info(f"val loss MAE: {loss_mae}")

                for i, key in enumerate(batch['key']):
                    results.append({
                        "key": key.item() if isinstance(key, torch.Tensor) else key,
                        "age": labels[i].item(),
                        f"pred_{organ}":y[i].item()
                    })

        running_loss_val += loss
        running_loss_mae += loss_mae
    logging.info(f"running_loss_val: {running_loss_val/len(ds_test)}")
    logging.info(f"running_loss_mae: {running_loss_mae/len(ds_test)}")

    if two_label == True:
        logging.info(f"running_loss_t: {running_loss_t/len(ds_test)}")
        logging.info(f"running_loss_i: {running_loss_i/len(ds_test)}")
        logging.info(f"running_loss_t_mae: {running_loss_t_mae/len(ds_test)}")
        logging.info(f"running_loss_i_mae: {running_loss_i_mae/len(ds_test)}")


    df = pd.DataFrame(results)
    if save_paths is None:
        save_path = "prediction.csv"
    else:
        save_path=f"{save_paths['results']}/prediction.csv"

    file_exists = os.path.isfile(save_path)
    df.to_csv(save_path, index=False, mode="a", header=not file_exists)
    logging.info(f"Predictions saved to {save_path}")

    
    return

@gin.configurable
def test_chkpt(model, ds_test, device, run_paths, data_type, feature_space = False, use_wandb = False):
    """This method is used to check if the right model checkpoint was loaded by creating a plot of some projections"""
    model.eval()
    embeddings_list = []
    labels_list = []
    types_list = []
    j = 0
    with torch.no_grad():  
        for batch in ds_test:  
            if j > 40:
                break  
            labels = batch['label'].view(-1, 1)
            with autocast(device_type=device.type):  
                if feature_space == True:  
                    _, z = model(batch)
                else:
                    z, _ = model(batch)
            gen_lists(z, labels, embeddings_list, labels_list, types_list, data_type)
            j+=1
        
        plot_results(embeddings_list, labels_list, types_list, 0, run_paths, feature_space = feature_space, use_wandb = use_wandb)


@gin.configurable
def test_chkpt_compare(model1, model2, ds_test, device, run_paths, data_type, use_wandb = False):
    """This method is used to check if the right model checkpoints were loaded by creating a plot of some projections from two models"""
    model1.eval()
    model2.eval()

    embeddings_list = []
    labels_list = []
    types_list = []
    sources_list = []

    j = 0
    with torch.no_grad():  
        for batch in ds_test:  
            if j > 40:
                break  
            labels = batch['label'].view(-1, 1)     

            for model, source in [(model1, 'model1'), (model2, 'model2')]:
                z, _ = model(batch)
                gen_lists(z, labels, embeddings_list, labels_list, types_list, data_type)
                sources_list.extend([source] * (len(z)* 2))
        
        plot_results(embeddings_list, labels_list, types_list, 0, run_paths, sources_list, True, use_wandb = use_wandb)
        