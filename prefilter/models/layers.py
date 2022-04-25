import torch
import math
import torch.nn as nn
import prefilter.utils as utils

__all__ = ["ResNet"]


class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class ResConv(torch.nn.Module):
    def __init__(self):
        pass
