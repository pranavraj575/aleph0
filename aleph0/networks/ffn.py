"""
simple feed forward NN
"""
import torch
from torch import nn


class FFN(nn.Module):
    """
    simple feed forward network with ReLU activation and specified hidden layers
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers=None,
                 activation=nn.ReLU,
                 device=None,
                 ):
        super().__init__()
        self.device = device
        if hidden_layers is None:
            hidden_layers = []
        self.nn_layers = nn.ModuleList()
        hidden_layers = [input_dim] + list(hidden_layers)
        for i in range(len(hidden_layers) - 1):
            self.nn_layers.append(nn.Linear(hidden_layers[i],
                                            hidden_layers[i + 1],
                                            device=self.device,
                                            ))
            self.nn_layers.append(activation())
        self.nn_layers.append(nn.Linear(hidden_layers[-1],
                                        output_dim,
                                        device=self.device,
                                        ))

    def forward(self, X):
        """
        :param X: (*, input_dim)
        :return: (*, output_dim)
        """
        for layer in self.nn_layers:
            X = layer(X)
        return X
