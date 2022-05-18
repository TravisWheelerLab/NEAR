import pdb

import torch
import math
import torch.nn as nn
import prefilter.utils as utils

__all__ = ["ResConv"]


class ResConv(torch.nn.Module):
    def __init__(self, filters, kernel_size, padding):
        super(ResConv, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.conv1 = torch.nn.Conv1d(filters, filters, kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm1d(filters)
        self.act = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(filters, filters, kernel_size, padding=padding)
        self.bn2 = torch.nn.BatchNorm1d(filters)

    def masked_forward(self, features, mask):
        x = self.conv1(features)
        mask = utils.mask_mask(mask)
        x = self.act(x)
        x = ~mask[:, None, :] * x
        x = self.conv2(x)
        mask = utils.mask_mask(mask)
        x = self.act(x)
        if self.padding == "valid":
            # two convolutions; so multiply half the kernel width by 2.
            # clearer than just self.kernel_size.
            features = features[
                :, :, 2 * (self.kernel_size // 2) : -2 * (self.kernel_size // 2)
            ]
        x = ~mask[:, None, :] * x
        return x + features, mask

    def forward(self, features):
        x = self.conv1(features)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        if self.padding == "valid":
            # two convolutions; so multiply half the kernel width by 2.
            # clearer than just self.kernel_size.
            features = features[
                :, :, 2 * (self.kernel_size // 2) : -2 * (self.kernel_size // 2)
            ]
        return x + features
