import torch.nn as nn
import torch.nn.functional as F


class DynamicEvalModel(nn.Module):
    def __init__(self, num_squares, other_features, num_hidden_layers=2):
        super(DynamicEvalModel, self).__init__()
        self.input_dim = num_squares + other_features
        self.hidden_dim = 128
        self.num_hidden_layers = num_hidden_layers

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim)
             for _ in range(self.num_hidden_layers)]
        )
        self.output_layer = nn.Linear(self.hidden_dim, num_squares)

    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = F.leaky_relu(layer(x))

        x = self.output_layer(x)

        return x
