"""File for multimodal model"""
import torch
import torch.nn as nn

from models.layers import ProjectionHead, TabularEncoder, PredictionHead
from models.ResNet3D import generate_ResNet
from torch.amp import autocast

import logging



class Multimodal(nn.Module):
    """
    This class is used as multimodal model.
    """
    def __init__(self, config, device, tabular_input_size, image_input_channels):
        super(Multimodal, self).__init__()
        if config.image_encoder_model_depth in [50, 101, 152, 200]:
            embedding_dim_image = 2048
        elif config.image_encoder_model_depth in [10, 18, 34]:
            embedding_dim_image = 512
        else:
            raise Exception("No such model depth available")

        self.mode = "contrastive"
        self.use_ph = config.use_ph
        self.device = device
        

        self.encoder_imaging = generate_ResNet(model_depth = config.image_encoder_model_depth, n_input_channels = image_input_channels)
        self.projector_imaging = ProjectionHead(config, embedding_dim_image, config.hidden_dim_image_ph)

        self.encoder_tabular = TabularEncoder(config, tabular_input_size)
        self.projector_tabular = ProjectionHead(config, config.embedding_dim_tabular, config.hidden_dim_tabular_ph)
        self.extra_bit = config.extra_bit

        if self.use_ph:
            if config.two_label == True:
                extra_bit = 0
                if self.extra_bit == True:
                    extra_bit = 1
                self.prediction_head = PredictionHead(config.projection_dim + extra_bit, config)
            else:
                self.prediction_head = PredictionHead(config.projection_dim * 2, config)
        else:
            if config.two_label == True:
                if embedding_dim_image == config.embedding_dim_tabular:
                    extra_bit = 0
                    if self.extra_bit == True:
                        extra_bit = 1
                    self.prediction_head = PredictionHead(embedding_dim_image + extra_bit, config)
                else:
                    raise ValueError(f"Embedding dimensions from encoders have to be the same")
            else:
                self.prediction_head = PredictionHead(embedding_dim_image + config.embedding_dim_tabular, config)

        

    def set_mode(self, mode):
        """
        This method can change the mode of the model between prediction and contrastive.
        """
        self.mode = mode

        if mode == "prediction":
            for p in self.encoder_imaging.parameters():
                p.requires_grad = False
            for p in self.encoder_tabular.parameters():
                p.requires_grad = False
            if self.use_ph:
                for p in self.projector_imaging.parameters():
                    p.requires_grad = False
                for p in self.projector_tabular.parameters():
                    p.requires_grad = False

    def freeze_encoder(self):
        """
        This method freezes the encoder.
        """
        for p in self.encoder_imaging.parameters():
            p.requires_grad = False
        for p in self.encoder_tabular.parameters():
            p.requires_grad = False
        if self.use_ph:
            for p in self.projector_imaging.parameters():
                p.requires_grad = False
            for p in self.projector_tabular.parameters():
                p.requires_grad = False

    def forward(self, batch, modality = "multimodal"):
        """
        Forward method for model.
        """
        if modality != "multimodal" and self.mode == "contrastive":
            raise ValueError(f"modality {modality} not available for contrastive pre-training")

        with autocast(device_type=self.device.type):
            if modality == "multimodal":
                y_i = self.encoder_imaging(batch['image'])
                y_t = self.encoder_tabular(batch['tabular_data'])
            elif modality == "image":
                y = self.encoder_imaging(batch['image'])
            elif modality == "tabular":
                y = self.encoder_tabular(batch['tabular_data'])
            else:
                raise ValueError(f"Unknown modality {modality}")


            if self.mode == "contrastive":
                # will never be called with modaility != multimodal
                z_i = self.projector_imaging(y_i)
                z_t = self.projector_tabular(y_t)
                z = torch.stack([z_i, z_t], dim=1)
                y = torch.stack([y_i, y_t], dim=1)
                return z, y
            

            elif self.mode == "prediction":
                if self.use_ph:
                    if modality == "multimodal":
                        y_i = self.projector_imaging(y_i)
                        y_t = self.projector_tabular(y_t)
                    elif modality == "image":
                        y = self.projector_imaging(y)
                        if self.extra_bit == True:
                            ones = torch.ones(y.size(0), 1, device=y.device, dtype=y.dtype)
                            y = torch.cat([y, ones], dim=1)

                    elif modality == "tabular":
                        y = self.projector_tabular(y)
                        if self.extra_bit == True:
                            zeros = torch.zeros(y.size(0), 1, device=y.device, dtype=y.dtype)
                            y = torch.cat([y, zeros], dim=1)

                if modality == "multimodal":
                    y = torch.cat((y_i, y_t), dim=1)
                output = self.prediction_head(y)
                return output



