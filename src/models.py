# Standard library imports
import os
import random
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Tuple
from pathlib import Path

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F


class ResConv(nn.Module):
    def __init__(self, in_dims, h_dims, kernel_size, h_kernel_size=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dims, h_dims, kernel_size, bias=True, padding='same', padding_mode='zeros')
        self.act = nn.ELU()
        self.conv2 = nn.Conv1d(h_dims, in_dims, h_kernel_size, bias=True, padding='same', padding_mode='zeros')
        #self.bn = nn.BatchNorm1d(in_dims)
    
    # expects [batch_size, in_dims, seq_len]
    def forward(self, x_in):
        x = self.act(self.conv1(self.act(x_in)))
        x = self.conv2(x)
        return x + x_in

class NEARResNet(nn.Module):
    def __init__(self, embedding_dim: int, num_layers: int = 5, kernel_size: int = 5, h_kernel_size: int = 1, in_symbols: int = 25):
        super().__init__()
        
        self.register_buffer('in_symbols', torch.tensor(in_symbols, dtype=torch.int32))

        self.embedding_layer = nn.Linear(in_symbols, embedding_dim, bias=False)
        self.conv_layers = nn.Sequential(*[ResConv(embedding_dim, embedding_dim, kernel_size, h_kernel_size=h_kernel_size) for _ in range(num_layers)])


    def embed(self, x):
        if len(x.shape) == 2:
            x = F.one_hot(x, num_classes=self.in_symbols.item()).float()
        assert(len(x.shape) == 3)
        x = self.embedding_layer(x)
        x = x.transpose(1, -1)

        return x


    # expects [batch_size, seq_len]
    # or      [batch_size, seq_len, in_symbols]
    def forward(self, x):
        x = self.embed(x)
        x = self.conv_layers(x)
        return x

class NEARUNet(nn.Module):
    def __init__(self, embedding_dim: int, in_symbols: int = 25, depth: int = 4, res_layers: int = 1, kernel_size: int = 5, h_kernel_size: int = 1, pool_size: int = 2):
        super().__init__()
        
        self.register_buffer('in_symbols', torch.tensor(in_symbols, dtype=torch.int32))

        self.embedding_layer = nn.Linear(in_symbols, embedding_dim, bias=False)
        self.conv_layers = nn.ModuleList([nn.Sequential(
            *[ResConv(embedding_dim, 
                      embedding_dim, 
                      kernel_size, 
                      h_kernel_size=h_kernel_size,) for _ in range(res_layers)]) for _ in range(depth)])
        self.max_pool = nn.MaxPool1d(2)
        
        self.deconv_layers = nn.ModuleList([nn.Sequential(nn.ConvTranspose1d(embedding_dim, 
                                                                             embedding_dim, 
                                                                             2, 
                                                                             stride=2),
            *[ResConv(embedding_dim, 
                      embedding_dim, 
                      kernel_size, 
                      h_kernel_size=h_kernel_size,) for _ in range(res_layers)]) for _ in range(depth)])
        self.last_layer = ResConv(embedding_dim, embedding_dim, kernel_size)


    def embed(self, x):
        if len(x.shape) == 2:
            x = F.one_hot(x, num_classes=self.in_symbols.item()).float()
        assert(len(x.shape) == 3)
        x = self.embedding_layer(x)
        x = x.transpose(1, -1)

        return x



    # expects [batch_size, seq_len]
    # or      [batch_size, seq_len, in_symbols]
    def forward(self, x):
        x = self.embed(x)
        x_stack = []
        for layer in self.conv_layers:
            x = layer(x)
            x_stack.append(x)
            x = self.max_pool(x)

        for layer in self.deconv_layers:
            x = layer(x)
            px = x_stack.pop()
            x = x + px

        x = self.last_layer(x)
        return x
