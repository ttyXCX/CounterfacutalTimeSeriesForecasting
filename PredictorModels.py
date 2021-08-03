import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_length, output_length, hidden_size=[256, 128]):
        super(MLP, self).__init__()

        layers = []
        for i in range(len(hidden_size)):  # loop hidden dimensions
            if i == 0:
                layers.append(nn.Linear(input_length, hidden_size[i]))
            else:
                layers.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size[-1], output_length))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# TODO
# test function
if __name__ == "__main__":
    data = torch.randn((100, 96))
    mlp = MLP(input_length=96, output_length=12, hidden_size=[64, 32])
    y_hat = mlp.forward(data)
