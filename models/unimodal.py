"""File for unimodal interface"""
import torch
import torch.nn as nn
from torch.amp import autocast
from abc import ABC, abstractmethod

from models.layers import ProjectionHead, PredictionHead


class Unimodal(nn.Module, ABC):
    """
    Unimodal abstract class that can be used as interface for unimodal models.
    """
    def __init__(self, config, device):
        super(Unimodal, self).__init__()
        self.mode = "contrastive"
        self.use_ph = config.use_ph
        self.device = device

        self.encoder = None

        self.projector = None
        self.prediction_head = None

    def init_heads(self, config, embedding_dim, hidden_dim_ph):
        """
        This method initializes the projection and prediction heads.
        """
        self.projector = ProjectionHead(config, embedding_dim, hidden_dim_ph)
        if self.use_ph:
            self.prediction_head = PredictionHead(config.projection_dim, config)
        else:
            self.prediction_head = PredictionHead(embedding_dim, config)

    def set_mode(self, mode):
        """
        This method can change the mode of the model between prediction and contrastive.
        """
        self.mode = mode
        if mode == "prediction":
            for p in self.encoder.parameters():
                p.requires_grad = False
            if self.use_ph and self.projector is not None:
                for p in self.projector.parameters():
                    p.requires_grad = False

    def forward(self, batch):
        """
        Forward method for model.
        """
        data = self.get_data(batch)
        with autocast(device_type=self.device.type):
            y = self.encoder(data)

            if self.mode == "contrastive":
                z = self.projector(y)
                return z, y
            elif self.mode == "prediction":
                if self.use_ph:
                    y = self.projector(y)
                return self.prediction_head(y)

    def freeze_encoder(self):
        """
        This method freezes the encoder.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.use_ph and self.projector is not None:
            for p in self.projector.parameters():
                p.requires_grad = False

    @abstractmethod
    def get_data(self, batch):
        pass