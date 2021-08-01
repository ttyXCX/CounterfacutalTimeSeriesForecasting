import torch
from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_length, output_length):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_length, output_length)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        y_hat = self.sigmoid(x)
        return y_hat