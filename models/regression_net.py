import torch
from torch import nn
import numpy as np


class RegressionNet(nn.Module):
    """

    A fully-connected network with 'num_hidden' hidden layers of 'hidden_dim' neurons each,\
    alternated with instances of 'nonlinearity'.

    """

    def __init__(self, input_shape, output_size, hidden_dim=40, num_hidden=4,
                 dropout=0.0, nonlinearity="Tanh", init_type=None, layer_norm=False):
        super(type(self), self).__init__()
        input_size = int(np.prod(input_shape))
        if num_hidden == 0:
            self.layers = nn.ModuleList([
                apply_init(nn.Linear(input_size, output_size), init_type=init_type, nonlinearity=nonlinearity),
            ])
        else:
            self.layers = nn.ModuleList([])
            self.layers.append(self.apply_init(nn.Linear(input_size, hidden_dim), init_type=init_type, nonlinearity=nonlinearity))
            if layer_norm:
                self.layers.append(nn.LayerNorm(normalized_shape=hidden_dim))
            self.layers.append(getattr(nn, nonlinearity)())
            self.layers.append(nn.Dropout(p=dropout))
            for _ in range(num_hidden - 1):
                self.layers.append(self.apply_init(nn.Linear(hidden_dim, hidden_dim), init_type=init_type, nonlinearity=nonlinearity))
                if layer_norm:
                    self.layers.append(nn.LayerNorm(normalized_shape=hidden_dim))
                self.layers.append(getattr(nn, nonlinearity)())
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(self.apply_init(nn.Linear(hidden_dim, output_size), init_type=init_type, nonlinearity=nonlinearity))

    def apply_init(self, layer, init_type="normal", nonlinearity="Tanh"):
        if init_type == "normal":
            nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain(nonlinearity.lower()))
        elif init_type == "normal":
            nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain(nonlinearity.lower()))
        elif init_type == "orthogonal":
            nn.init.orthogonal_(layer.weight, nn.init.calculate_gain(nonlinearity.lower()))

        nn.init.constant_(layer.bias, 0.0)
        return layer

    def forward(self, X):
        """
        Forward pass through the network. Returns logits.
        Expected input shape: [batch_size, *self.input_shape]
        Output shape: [batch_size, output_size].
        """
        for layer in self.layers:
            X = layer(X)
        return X