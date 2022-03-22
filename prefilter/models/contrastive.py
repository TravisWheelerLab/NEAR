import torch
import torch.nn as nn

from prefilter.models import ResidualBlock
from prefilter.utils import PROT_ALPHABET

__all__ = ["ResNet1d"]


class ResNet1d(torch.nn.Module):
    def __init__(self):
        super(ResNet1d, self).__init__()
        self.vocab_size = len(PROT_ALPHABET)
        self.res_block_n_filters = 256
        self.feat_dim = 128
        self.res_block_kernel_size = 3
        self.n_res_blocks = 18
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
        self.projection = nn.Linear(self.res_block_n_filters, self.feat_dim)

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
        x = x.mean(axis=-1)
        return self.projection(x)


if __name__ == "__main__":
    # tests
    convnet_1d = ResNet1d()

    data_1d = torch.rand((32, len(PROT_ALPHABET), 105))

    print(convnet_1d(data_1d).shape)
