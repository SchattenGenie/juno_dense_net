import torch
from torch import nn

import numpy as np

class RegressionNet(nn.Module):
    """

    A fully-connected network with 'num_hidden' hidden layers of 'hidden_dim' neurons each,\
    alternated with instances of 'nonlinearity'.

    """

    def __init__(self, input_shape, output_size, hidden_dim=40, num_hidden=4, nonlinearity="Tanh", init_type=None):
        super(type(self), self).__init__()
        input_size = int(np.prod(input_shape))
        if num_hidden == 0:
            self.layers = nn.ModuleList([
                nn.Linear(input_size, output_size),
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Linear(input_size, hidden_dim),
                getattr(nn, nonlinearity)()
            ])
            for _ in range(num_hidden - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(getattr(nn, nonlinearity)())
            self.layers.append(nn.Linear(hidden_dim, output_size))

    def forward(self, X):
        """
        Forward pass through the network. Returns logits.
        Expected input shape: [batch_size, *self.input_shape]
        Output shape: [batch_size, output_size].
        """
        for layer in self.layers:
            X = layer(X)
        return X