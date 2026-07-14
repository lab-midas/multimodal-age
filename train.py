"""This file does the training of the model"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import wandb
import gin
from losses.KernelizedLoss import KernelizedLoss
from utils.utils_disp import display, log_projection, plot_start, plot_results, gen_lists
import math

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from torch.amp import autocast, GradScaler
import torch.nn.functional as F

@gin.configurable
class Trainer(object):
    """
    Trainer class
    """
    def __init__(self, model, ds_train, ds_val, run_paths, config, data_type, use_wandb, device, ckpt_interval, ckpt_interval_ft):

        # Loss objective
        self.data_type = data_type
        if self.data_type == "multimodal":
            self.loss_object = KernelizedLoss(method = config.loss_method, kernel = self.gaussian_kernel, temperature = 0.07, base_temperature = 0.07, loss_mode = config.loss_mode, larger_matrix = config.larger_matrix, split_matrix = config.split_matrix, cross_weight = config.cross_weight).to(device)
        else:
            self.loss_object = KernelizedLoss(method = config.loss_method, kernel = self.gaussian_kernel, temperature = 0.07, base_temperature = 0.07, loss_mode = config.loss_mode).to(device)
          
        self.loss_object_finetune = nn.MSELoss().to(device)
        self.loss_object_MAE = nn.L1Loss().to(device)
        # self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.optimizer_finetune = optim.Adam(model.parameters(), lr=config.learning_rate_finetune)
        # self.scheduler = ExponentialLR(self.optimizer, gamma=config.decay_rate)
        self.scheduler_finetune = ExponentialLR(self.optimizer_finetune, gamma=config.decay_rate_finetune)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=config.decay_rate)
        # self.scheduler_finetune = torch.optim.lr_scheduler.StepLR(self.optimizer_finetune, step_size=1, gamma=0.1)
        # Checkpoint Manager

        # Metrics

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.epochs = config.total_epochs
        self.epochs_finetune = config.total_epochs_finetune
        self.ckpt_interval = ckpt_interval
        self.ckpt_interval_ft = ckpt_interval_ft
        self.use_wandb = use_wandb
        self.device = device
        self.additional_loss = config.additional_loss
        if self.data_type == "multimodal":
            self.two_label = config.two_label
        else:
            self.two_label = False

        self.log_template = 'Epoch: {}, Loss: {}, Validation Loss: {}'
        self.log_template_finetune = 'Epoch: {}, Loss FT: {}, Validation Loss FT: {}'

        self.scaler = GradScaler()
        self.data_type = data_type
        self.sigma = config.sigma

    def gaussian_kernel(self, x, y):
        """
        This method calculates the kernel based on the label difference.
        """
        x = x - y.T
        return torch.exp(-(x**2) / (2*(self.sigma**2))) / (math.sqrt(2*torch.pi)*self.sigma)

    def train(self):
        """
        This method traines the model.
        """
        torch.cuda.empty_cache()
        plot_start(self.model, self.ds_val, self.device, self.run_paths, self.data_type, self.two_label, use_wandb = self.use_wandb)
        best_loss = math.inf
        for epoch in range(self.epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Current learning rate: {current_lr}")
            # wandb.log({"learning_rate": current_lr}, epoch)
            loss = self.train_epoch(epoch, False)
            if torch.isnan(torch.tensor(loss)):
                return
            val_loss = self.validate(epoch + 1)
            self.log(epoch + 1, loss, val_loss, self.log_template)
            if self.use_wandb:
                wandb.log({"loss": loss, "val_loss": val_loss, "learning rate": current_lr}) 
            if (epoch + 1) % self.ckpt_interval == 0:
                if val_loss < best_loss:
                    self.safe_checkpoint(path = self.run_paths["path_ckpts_train"])
                    best_loss = val_loss
            self.scheduler.step()
            
        logging.info(f'Training finished after {self.epochs} epochs')
        if self.epochs % self.ckpt_interval != 0:
            logging.info(f'Saving final checkpoint')
            torch.save(self.model.state_dict(), self.run_paths['path_ckpts_train'])
    

    def train_epoch(self, epoch, finetune):
        """
        This method executes one epoch.
        """
        self.model.train(True)
        running_loss = 0.0
        
        for i, batch in enumerate(self.ds_train):
            if self.use_wandb:
                if self.data_type == "multimodal" or self.data_type == "image":
                    if i % (len(self.ds_train)//5) == 0:
                        self.log_samples(batch, epoch, i)

            if finetune == True:
                loss, loss_mae = self.finetune_step(batch)
                current_lr = self.optimizer_finetune.param_groups[0]['lr']
                logging.info(f"Current learning rate: {current_lr}")
                logging.info(f"loss MAE: {loss_mae}")
                self.scheduler_finetune.step()
            else:
                loss = self.train_step(batch, epoch)
            if torch.isnan(torch.tensor(loss)):
                return loss
            logging.info(f"loss: {loss}")
            running_loss += loss
            torch.cuda.empty_cache()

        return running_loss / len(self.ds_train)


    def train_step(self, batch, epoch):
        """
        This method makes one trainingstep.
        """
        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type):
            z, y = self.model(batch)
            if self.data_type == "multimodal" and self.two_label == True:
                loss = self.loss_object(z, batch['label'].view(-1, 1).to(self.device), batch['label-t'].view(-1,1).to(self.device))
            else:
                loss = self.loss_object(z, batch['label'].view(-1, 1).to(self.device))
    
        if z.ndim == 3:
            z_i, z_t = z[:, 0, :], z[:, 1, :]
            loss = self.apply_alignment_loss(z_i, z_t, loss, epoch)

        if torch.isnan(loss):
            self.check_error(z, y)
    
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # loss.backward()
        # self.optimizer.step()

        return loss.item()


    def log_samples(self, batch, epoch, batch_idx):
        """
        This method logs samples to wandb.
        """
        img = batch['image']
        img = img[0,0,:,:,int(img.size()[3]//2)].cpu().numpy()
        plt.imshow(img, cmap='gray')
        if self.use_wandb:
            wandb.log({f"samples/epoch_{epoch}_sample_{batch_idx}": wandb.Image(plt)}, commit=True)


    def apply_alignment_loss(self, z_i, z_t, loss, epoch):
        """
        This method adds an alignment loss if needed.
        """
        if self.additional_loss == "L2 and L1":
            if epoch < 2:
                alignment_loss = (F.mse_loss(z_i, z_t))
            else: 
                alignment_loss = (F.l1_loss(z_i, z_t))
            logging.info(f"alignment loss: {alignment_loss}")
            loss = loss + alignment_loss

        
        elif self.additional_loss == "L2":
            alignment_loss = (F.mse_loss(z_i, z_t))
            logging.info(f"alignment loss: {alignment_loss}")
            loss = loss + alignment_loss

        elif self.additional_loss == "L1":
            alignment_loss = (F.l1_loss(z_i, z_t))
            loss = loss + alignment_loss
        
        return loss

    def check_error(self, z, y):
        """
        This method checks if any projections contains a NAN.
        """
        logging.info(f"Loss is NaN! Skipping backward and optimizer step.")
        if z.ndim == 3:
            z_i, z_t = z[:, 0, :], z[:, 1, :]
            y_i, y_t = y[:, 0, :], y[:, 1, :]
            if torch.isnan(z_i).any():
                logging.info(f"Image Projection contains NaNs!")
            if torch.isnan(z_t).any():
                logging.info(f"Tabular Projection contains NaNs!")
            if torch.isnan(y_i).any():
                logging.info(f"Image Embedding contains NaNs!")
            if torch.isnan(y_t).any():
                logging.info(f"Tabular Embedding contains NaNs!")
        
        else:
            if torch.isnan(z).any():
                logging.info(f"Projection contains NaNs!")
            if torch.isnan(y).any():
                logging.info(f"Embedding contains NaNs!")

        
    def validate(self, epoch):
        """
        This method validates the model in the pre-training.
        """
        self.model.eval()
        running_loss = 0.0
        embeddings_list = []
        labels_list = []
        types_list = []
        j = 0
        with torch.no_grad():  
            for batch in self.ds_val:   
                labels = batch['label'].view(-1, 1).to(self.device)
                with autocast(device_type=self.device.type):     
                    z, y = self.model(batch)
                    if self.data_type == "multimodal" and self.two_label == True:
                        loss = self.loss_object(z, labels, batch['label-t'].view(-1,1).to(self.device))
                    else:
                        loss = self.loss_object(z, labels)
                    
                if j < 40:
                    if self.data_type == "multimodal" and self.two_label == True:
                        labels_t = batch['label-t'].view(-1,1).to(self.device)
                        gen_lists(z, labels, embeddings_list, labels_list, types_list, self.data_type, labels_t)
                    else:
                        gen_lists(z, labels, embeddings_list, labels_list, types_list, self.data_type)


                running_loss += loss.item()
                j+=1
            
            plot_results(embeddings_list, labels_list, types_list, epoch, self.run_paths, use_wandb = self.use_wandb)

            return running_loss / len(self.ds_val)


    def safe_checkpoint(self, path):
        """
        This method saves a model.
        """
        logging.info(f'Saving checkpoint to {path}.')
        torch.save(self.model.state_dict(), path)

    def finetune(self):
        """
        This method fine-tunes a model.
        """
        logging.info(f"Start Finetuning")
        j = 0
        best_mae = math.inf
        for epoch in range(self.epochs_finetune):
            loss = self.train_epoch(epoch, True)
            val_loss, val_loss_mae = self.finetune_validate()

            self.log(epoch + 1, loss, val_loss, self.log_template_finetune)
            logging.info(f"val loss mae: {val_loss_mae}")
            if self.use_wandb:
                wandb.log({"loss ft": loss, "val loss ft": val_loss, "val loss mae": val_loss_mae}) 
            if (epoch + 1) % self.ckpt_interval_ft == 0:
                if val_loss_mae < best_mae:
                    self.safe_checkpoint(path = self.run_paths["path_ckpts_train_ft"])
                    best_mae = val_loss_mae

        logging.info(f'Saving final checkpoint')
        torch.save(self.model.state_dict(), self.run_paths['path_ckpts_train_ft'])

    def finetune_validate(self):
        """
        This method validates the model in the fine-tuning.
        """
        self.model.eval()
        running_loss_val = 0.0
        running_loss_mae = 0.0
        for batch in self.ds_val:
            with autocast(device_type=self.device.type):
                if self.two_label == True:
                    y_i = self.model(batch, modality="image")
                    labels_i = batch['label'].view(-1, 1).to(self.device)
                    loss_i = self.loss_object_finetune(y_i, labels_i)
                    loss_mae_i = self.loss_object_MAE(y_i, labels_i)

                    labels_t = batch['label-t'].view(-1, 1).to(self.device)
                    y_t = self.model(batch, modality="tabular")
                    loss_t = self.loss_object_finetune(y_t, labels_t)
                    loss_mae_t = self.loss_object_MAE(y_t, labels_t)

                    loss = (loss_i + loss_t) / 2
                    loss_mae = (loss_mae_i + loss_mae_t) / 2

                    logging.info(f"val loss: {loss}")
                    logging.info(f"val loss MAE: {loss_mae}")
                    
                else:
                    labels = batch['label'].view(-1, 1).to(self.device)
                    y = self.model(batch)
                    loss = self.loss_object_finetune(y, labels)
                    loss_mae = self.loss_object_MAE(y, labels)
                    logging.info(f"val loss: {loss}")
                    logging.info(f"val loss MAE: {loss_mae}")

            if torch.isnan(torch.tensor(loss)):
                return loss
            running_loss_val += loss
            running_loss_mae += loss_mae
        return running_loss_val / len(self.ds_val), running_loss_mae / len(self.ds_val)


    def finetune_step(self, batch):
        """
        This method makes one fine-tuningstep.
        """
        self.optimizer_finetune.zero_grad()
        # logging.info(f"labels : {labels}")

        with autocast(device_type=self.device.type):
            if self.two_label == True:
                y_i = self.model(batch, modality="image")
                labels_i = batch['label'].view(-1, 1).to(self.device)
                loss_i = self.loss_object_finetune(y_i, labels_i)
                loss_mae_i = self.loss_object_MAE(y_i, labels_i)

                labels_t = batch['label-t'].view(-1, 1).to(self.device)
                y_t = self.model(batch, modality="tabular")
                loss_t = self.loss_object_finetune(y_t, labels_t)
                loss_mae_t = self.loss_object_MAE(y_t, labels_t)

                loss = (loss_i + loss_t) / 2
                loss_mae = (loss_mae_i + loss_mae_t) / 2

            else:
                labels = batch['label'].view(-1, 1).to(self.device)
                y = self.model(batch)
                # logging.info(f"prediction : {y}")
                loss = self.loss_object_finetune(y, labels)
                loss_mae = self.loss_object_MAE(y, labels)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer_finetune)
        self.scaler.update()

        return loss.item(), loss_mae.item()

    def log(self, epoch, loss, val_loss, template):
        """
        This method logs losses over the epochs.
        """
        logging.info(template.format(epoch,
            loss,
            val_loss))
            
            

