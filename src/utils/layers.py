import math

import torch
import torch.nn as nn

import src.utils as utils

__all__ = ["ResConv2d", "PositionalEncoding", "ResConv"]


class ResConv(torch.nn.Module):
    def __init__(self, filters, kernel_size, padding, padding_mode):
        super().__init__()

        self.padding = padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.conv1 = torch.nn.Conv1d(
            filters,
            filters,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.act = torch.nn.ELU()
        self.conv2 = torch.nn.Conv1d(
            filters,
            filters,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

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
                :,
                :,
                2 * (self.kernel_size // 2) : -2 * (self.kernel_size // 2),
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
                :,
                :,
                2 * (self.kernel_size // 2) : -2 * (self.kernel_size // 2),
            ]
        return x + features


class ResConv2d(torch.nn.Module):
    def __init__(self, filters, kernel_size, padding, padding_mode):
        super().__init__()

        self.padding = padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.conv1 = torch.nn.Conv2d(
            filters,
            filters,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        self.act = torch.nn.ELU()
        self.conv2 = torch.nn.Conv2d(
            filters,
            filters,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

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
                :,
                :,
                2 * (self.kernel_size // 2) : -2 * (self.kernel_size // 2),
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
                :,
                :,
                2 * (self.kernel_size // 2) : -2 * (self.kernel_size // 2),
            ]
        return x + features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
