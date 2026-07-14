"""File for tabular model"""
from models.layers import TabularEncoder
from models.unimodal import Unimodal

class Tabular(Unimodal):
    """
    Tabular unimodal model.
    """
    def __init__(self, config, device, tabular_input_size):
        super(Tabular, self).__init__(config, device)
        self.encoder = TabularEncoder(config, tabular_input_size)
        self.init_heads(config, config.embedding_dim_tabular, config.hidden_dim_tabular_ph)

    def get_data(self, batch):
        return batch["tabular_data"]



