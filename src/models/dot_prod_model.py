import pytorch_lightning as pl
import torch
import torch.nn as nn


class DotProdModel(pl.LightningModule):
    def __init__(self):
        super(DotProdModel, self).__init__()
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True)

    # expects [batch, dim, seq_len]
    def Dot(self, A, B, activation=torch.sigmoid):
        matrix = torch.einsum("bei,bej->...bij", A, B)
        if activation is None:
            return matrix
        return activation(matrix + self.bias)

    # expects [batch, dim, seq_len]
    def L2Dist(self, A, B):
        return torch.cdist(torch.transpose(A, -1, -2), torch.transpose(B, -1, -2))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        dims,
        num_layers,
        kernel_size,
        activation=nn.ELU,
        padding="same",
        padding_mode="circular",
        groups=1,
    ):
        super(ResidualBlock, self).__init__()

        layers = []

        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    dims,
                    dims,
                    kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    groups=groups,
                )
            )
            layers.append(activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.layers(x)
        return x


class ResNet(DotProdModel):
    def __init__(
        self,
        emb_dim,
        blocks,
        block_layers,
        start_emb=20,
        first_kernel=10,
        kernel_size=3,
        activation=nn.ELU,
        padding="same",
        padding_mode="circular",
        groups=1,
    ):
        super(ResNet, self).__init__()

        layers = [
            nn.Conv1d(
                start_emb,
                emb_dim,
                first_kernel,
                padding=padding,
                padding_mode=padding_mode,
            ),
            activation(),
        ]
        for i in range(blocks):
            layers.append(
                ResidualBlock(
                    emb_dim,
                    block_layers,
                    kernel_size,
                    activation,
                    padding=padding,
                    padding_mode=padding_mode,
                    groups=groups,
                )
            )
            # layers.append(nn.Dropout(0.1))

        layers.append(
            nn.Conv1d(emb_dim, emb_dim, 1, padding=padding, padding_mode=padding_mode)
        )
        layers.append(activation())
        layers.append(
            nn.Conv1d(emb_dim, emb_dim, 1, padding=padding, padding_mode=padding_mode)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
