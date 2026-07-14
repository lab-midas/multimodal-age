"""File for different model blocks and layers that can be used in mutiple models"""
import torch
import torch.nn as nn
import torch.nn.init as init

class ProjectionHead(nn.Module):
    """
    Class for projection head.
    """
    def __init__(self, config, input_dim, embedding_dim):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(input_dim, embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.relu = nn.LeakyReLU()  
        self.l2 = nn.Linear(embedding_dim, config.projection_dim)
        self.ln2 = nn.LayerNorm(config.projection_dim)

        init.kaiming_normal_(self.l1.weight, nonlinearity='leaky_relu')
        init.zeros_(self.l1.bias)

        init.kaiming_normal_(self.l2.weight, nonlinearity='leaky_relu')
        init.zeros_(self.l2.bias)


    def forward(self, x):
        x = self.l1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.ln2(x)
        return x

class TabularEncoder(nn.Module):
    """
    Class for tabular encoder.
    """
    def __init__(self, config, tabular_input_size):
        super(TabularEncoder, self).__init__()
        self.encoder = self.build_encoder(config, tabular_input_size)

    def build_encoder(self, config, tabular_input_size):
        modules = [nn.Linear(tabular_input_size, config.embedding_dim_tabular)]
        for _ in range(config.tabular_encoder_num_layers - 1):
            modules.extend([nn.BatchNorm1d(config.embedding_dim_tabular), nn.ReLU(), nn.Linear(config.embedding_dim_tabular, config.embedding_dim_tabular)])
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.encoder(x)
        return x

class PredictionHead(nn.Module):
    """
    Class for prediction head.
    """
    def __init__(self, in_dim, config):
        super(PredictionHead, self).__init__()

        layers = []
        input_dim = in_dim

        for hidden_dim in config.hidden_dims_prediction_head:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.head = nn.Sequential(*layers)


    def forward(self, x):
        return self.head(x)