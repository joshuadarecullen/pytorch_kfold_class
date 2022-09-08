import torch
from torch import nn


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Model
class Adversarial_LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Adversarial_LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.output = nn.Softmax(dim=1)
    def forward(self, x):
        return self.output(self.linear(x))
