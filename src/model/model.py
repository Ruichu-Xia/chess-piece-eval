import torch.nn as nn
import torch.nn.functional as F


class DynamicEvalModel(nn.Module):
    def __init__(self, num_squares,
                 other_features,
                 hidden_dim=128,
                 num_hidden_layers=2):

        super(DynamicEvalModel, self).__init__()
        self.num_squares = num_squares
        self.input_dim = num_squares + other_features
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_batchnorm = nn.BatchNorm1d(self.hidden_dim)

        self.hidden_layers = nn.ModuleList()
        self.hidden_batchnorms = nn.ModuleList()

        for _ in range(self.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(self.hidden_dim,
                                                self.hidden_dim))
            self.hidden_batchnorms.append(nn.BatchNorm1d(self.hidden_dim))

        self.output_layer = nn.Linear(self.hidden_dim, num_squares)

    def forward(self, x):
        mask = x[:, :self.num_squares]

        x = self.input_layer(x)
        x = self.input_batchnorm(x)
        x = F.leaky_relu(x)

        for layer, batchnorm in zip(self.hidden_layers,
                                    self.hidden_batchnorms):
            x = layer(x)
            x = batchnorm(x)
            x = F.leaky_relu(x)

        x = self.output_layer(x)
        x = F.relu(x)
        x = x * mask

        return x
