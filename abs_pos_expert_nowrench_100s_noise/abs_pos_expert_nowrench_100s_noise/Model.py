import torch
import torch.nn as nn
import torch.nn.functional as F

class FFBC(nn.Module):
    def __init__(self):
        super(FFBC, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(700, 1400).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(1400, 2100).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(2100, 4200).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(4200, 4200).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(4200, 2100).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(2100, 1400).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(1400, 1000).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(1000, 800).to(torch.float32))
        self.layers.append(nn.ReLU(inplace=False))
        self.layers.append(nn.Linear(800, 700).to(torch.float32))

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x