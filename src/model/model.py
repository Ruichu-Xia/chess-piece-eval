import torch.nn as nn
import torch.nn.functional as F


class DynamicEvalModel(nn.Module):
    def __init__(self, num_squares, other_features):
        super(DynamicEvalModel, self).__init__()
        self.input_dim = num_squares + other_features
        self.hidden_dim = 128

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, num_squares)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x
