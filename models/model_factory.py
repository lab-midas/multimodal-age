"""File for model factory to handle multiple models"""
import json
import gin
from models.multimodal import Multimodal
from models.tabular_model import Tabular
from models.image_model import Image
from types import SimpleNamespace
import torch
import torch.nn as nn

@gin.configurable
class ModelFactory():
    """
    This factory provides different dnn networks
    """
    def __init__(self, encoder_name, device):
        self.config = {}
        self.model = None
        self.encoder_name = encoder_name
        self.device = device
        self.tabular_input_size = None
        self.image_input_channels  = None
        self.encoder_dictionary = {
            "tabular": (Tabular, ["tabular_input_size"]),
            "image": (Image, ["image_input_channels"]),
            "multimodal": (Multimodal, ["tabular_input_size", "image_input_channels"])
        }
  
    
    def set_config(self, config):
        """
        This function is used to set the config parameters for the used model. 
        It must be called before the method self.get_model() is used
        """
        self.config = config

    def create_model(self):
        """
        This method returns the selectes model. Before calling this method you have to ensure that the config is set.
        """
        entry = self.encoder_dictionary.get(self.encoder_name)
        if entry is None:
            raise ValueError(f"Encoder '{self.encoder_name}' not valid")

        encoder, inputs = entry

        kwargs = {}
        for arg in inputs:
            value = getattr(self, arg, None)
            if value is None:
                raise ValueError(f"{arg} must be set before creating a {self.encoder_name} model")
            kwargs[arg] = value

        self.model = encoder(self.config, self.device, **kwargs)
        return self.model
    
    def get_model(self):
        return self.model

    
    def load_config(self, hyperparameter_tuning, use_wandb, config_path = None):
        """
        This method returns the config for the model and the training. 
        By using the parameter 'hyperparameter_tuning' it can be specified whether to load a config for hyperparameter optimization or a config for a single run. 
        With the parameter 'use_wandb' the type of the return value is defined.
        Use following name conventions for storing the configurations: 
        - Hyperparameter optimization: 'sweep_{self.encoder_name}_config.json' 
        - Single Run: 'single_run_{self.encoder_name}_config.json' 
        """
        if hyperparameter_tuning and use_wandb:
            if config_path == None: 
                config_path = f"./configs/sweep_{self.encoder_name}_config.json"
            config_data = {}
            with open(config_path, "r") as file:
                config_data = json.load(file)
        elif not hyperparameter_tuning:
            if config_path == None: 
                config_path = f"./configs/single_run_{self.encoder_name}_config.json"
            with open(config_path, "r") as f:
                config_data = json.load(f)
            if not use_wandb:
                config_data = SimpleNamespace(**config_data)
        else: 
            raise Exception("Usage of hyperparameter tuning without wandb is not possible")
        self.config = config_data
        return config_data
        
    def get_encoder_name(self):
        """
        This method returns the name of the used encoder.
        """
        return self.encoder_name

    def set_mode(self, mode):
        """
        This method sets the mode of the model.
        """
        self.model.set_mode(mode)

    def set_tabular_input_size(self, tabular_input_size):
        """
        This method sets the size of the first Dense layer in the tabular encoder.
        """
        self.tabular_input_size = tabular_input_size

    def get_tabular_input_size(self):
        """
        This method returns the amount of columns in the tabular data.
        """
        return self.tabular_input_size

    def set_image_input_channels(self, image_input_channels):
        """
        This method sets the input channels for the image encoder.
        """
        self.image_input_channels = image_input_channels

    def get_image_input_channels(self):
        """
        This method returns the input channels for the image encoder.
        """
        return self.image_input_channels
    