import torch
import torch.nn as nn

amino_n_to_a = [c for c in "ARNDCQEGHILKMFPSTWYVBZXJ*U"]
amino_a_to_n = {c: i for i, c in enumerate("ARNDCQEGHILKMFPSTWYVBZXJ*U")}

__all__ = [
    "amino_n_to_v",
    "ResidualBlock",
    "DotProdModel",
    "ResNet",
    "amino_a_to_n",
]

amino_frequencies = torch.tensor(
    [
        0.074,
        0.042,
        0.044,
        0.059,
        0.033,
        0.058,
        0.037,
        0.074,
        0.029,
        0.038,
        0.076,
        0.072,
        0.018,
        0.040,
        0.050,
        0.081,
        0.062,
        0.013,
        0.033,
        0.068,
    ]
)

amino_n_to_v = torch.zeros(len(amino_n_to_a), 20)
for i in range(20):
    amino_n_to_v[i, i] = 1.0

amino_n_to_v[amino_a_to_n["B"], amino_a_to_n["D"]] = 0.5
amino_n_to_v[amino_a_to_n["B"], amino_a_to_n["N"]] = 0.5

amino_n_to_v[amino_a_to_n["Z"], amino_a_to_n["Q"]] = 0.5
amino_n_to_v[amino_a_to_n["Z"], amino_a_to_n["E"]] = 0.5

amino_n_to_v[amino_a_to_n["J"], amino_a_to_n["I"]] = 0.5
amino_n_to_v[amino_a_to_n["J"], amino_a_to_n["L"]] = 0.5

amino_n_to_v[amino_a_to_n["X"]] = amino_frequencies
amino_n_to_v[amino_a_to_n["*"]] = amino_frequencies
amino_n_to_v[amino_a_to_n["U"]] = amino_frequencies


def L2Dist(A, B):
    return torch.cdist(torch.transpose(A, -1, -2), torch.transpose(B, -1, -2))


class DotProdModel(nn.Module):
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


# Residual block class
# Takes [batch_size, channels, seq_len] and outputs the same shape
# Performs (num_layers) 1d convolutions and activations and then
# adds the output to the original input
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

        layers.append(nn.Conv1d(emb_dim, emb_dim, 1, padding=padding, padding_mode=padding_mode))
        layers.append(activation())
        layers.append(nn.Conv1d(emb_dim, emb_dim, 1, padding=padding, padding_mode=padding_mode))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
