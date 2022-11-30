import torch
import torch.nn as nn


class Lights(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fcs = nn.Sequential(
            nn.LazyLinear(4),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        return self.fcs(x)