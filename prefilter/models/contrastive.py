import torch
import torch.nn as nn

from prefilter.models import ResidualBlock
from prefilter.utils import PROT_ALPHABET

__all__ = ["ResNet1d", "ResNet2d"]


class ResNet1d(torch.nn.Module):
    def __init__(self):
        super(ResNet1d, self).__init__()
        self.vocab_size = len(PROT_ALPHABET)
        self.res_block_n_filters = 256
        self.res_block_kernel_size = 3
        self.n_res_blocks = 6
        self.res_bottleneck_factor = 1

        self._setup_layers()

    def _setup_layers(self):

        self.initial_conv = nn.Conv1d(
            in_channels=self.vocab_size,
            out_channels=self.res_block_n_filters,
            kernel_size=self.res_block_kernel_size,
            padding="same",
        )

        self.embedding_trunk = torch.nn.ModuleList()

        for layer_index in range(self.n_res_blocks):
            self.embedding_trunk.append(
                ResidualBlock(
                    self.res_block_n_filters,
                    self.res_bottleneck_factor,
                    self.res_block_kernel_size,
                    layer_index,
                    1,
                    dilation_rate=None,
                )
            )

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        return x.mean(axis=-1)


class ResNet2d(torch.nn.Module):
    def __init__(self):
        super(ResNet2d, self).__init__()
        self.vocab_size = len(PROT_ALPHABET)
        self.res_block_n_filters = 256
        self.res_block_kernel_size = 3
        self.n_res_blocks = 6
        self.res_bottleneck_factor = 1

        self._setup_layers()

    def _setup_layers(self):

        self.initial_conv = nn.Conv2d(
            in_channels=self.vocab_size,
            out_channels=self.res_block_n_filters,
            kernel_size=self.res_block_kernel_size,
            padding="same",
        )

        self.embedding_trunk = torch.nn.ModuleList()

        for layer_index in range(self.n_res_blocks):
            self.embedding_trunk.append(
                ResidualBlock(
                    self.res_block_n_filters,
                    self.res_bottleneck_factor,
                    self.res_block_kernel_size,
                    layer_index,
                    1,
                    dilation_rate=None,
                    typ="2d",
                )
            )

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        # do GAP for lack of anything else...
        return x.mean(axis=-1).mean(axis=-1)


if __name__ == "__main__":
    # tests
    convnet_1d = ResNet1d()
    convnet_2d = ResNet2d()

    data_1d = torch.rand((32, len(PROT_ALPHABET), 105))
    data_2d = torch.rand((32, len(PROT_ALPHABET), 105, 30))

    print(convnet_1d(data_1d).shape)
    print(convnet_2d(data_2d).shape)
