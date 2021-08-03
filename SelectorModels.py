import torch
from torch import nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, input_length, output_length):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_length, output_length)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_out = self.linear(x)
        x_out = self.sigmoid(x_out)
        return x_out


# TODO
# test function
if __name__ == "__main__":
    data = torch.randn((100, 96))
    threshold = 0.5  # probability threshold
    logistic = LogisticRegression(input_length=96, output_length=12)
    y_hat = logistic.forward(data)
    y_se = (y_hat >= threshold) + 0
