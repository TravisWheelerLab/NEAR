import torch
from torch import nn
from torch.nn import functional as F


class NEARResConv(nn.Module):
    """A one-dimensional residual block.
    Each layer performs conv(dim->hdim), act(), conv(hdim->dim), skip connection

    Parameters
    ----------
    in_dims : int
        Number of input (and output) channels.
    h_dims : int
        Hidden channel width in the first convolution.
    kernel_size : int
        Size of the main convolution kernel.
    h_kernel_size : int
        Kernel size of the second (projection) convolution.
    """
    def __init__(self, in_dims: int,
                 h_dims: int,
                 kernel_size: int,
                 h_kernel_size: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_dims, h_dims, kernel_size, bias=True, padding='same', padding_mode='zeros')
        self.act = nn.ELU()
        self.conv2 = nn.Conv1d(h_dims, in_dims, h_kernel_size, bias=True, padding='same', padding_mode='zeros')
    
    # expects [batch_size, in_dims, seq_len]
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Parameters
        ----------
        x_in : torch.Tensor
            Input tensor of shape `[batch, in_dims, seq_len]`.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape and dtype as *x_in*.
        """

        x = self.act(self.conv1(self.act(x_in)))
        x = self.conv2(x)
        return x + x_in

class NEARResNet(nn.Module):
    def __init__(self, embedding_dim: int,
                 num_layers: int = 5,
                 kernel_size: int = 5,
                 h_kernel_size: int = 1,
                 in_symbols: int = 25):
        """Residual CNN used in the **NEAR** sequence-embedding model.

        A stack of `ResConv` blocks applied to one-hot or dense encoded
        biological sequences.

        Parameters
        ----------
        embedding_dim : int
            Width of the learned per-symbol embedding.
        num_layers : int
            Number of residual blocks.
        kernel_size : int
            Convolution kernel size in each block.
        h_kernel_size : int
            Kernel size of the projection convolution inside each block.
        in_symbols : int
            Alphabet size.
        """

        super().__init__()
        
        self.register_buffer('in_symbols', torch.tensor(in_symbols, dtype=torch.int32))

        self.embedding_layer = nn.Linear(in_symbols, embedding_dim, bias=False)
        self.conv_layers = nn.Sequential(*[NEARResConv(embedding_dim, embedding_dim, kernel_size, h_kernel_size=h_kernel_size) for _ in range(num_layers)])


    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """ Embed tokens or one-hot encoded sequence into initial representation.

               Parameters
               ----------
               x : torch.Tensor
                   Input tensor of shape `[batch, alphabet_size, seq_len]` or `[batch, seq_len]`.

               Returns
               -------
               torch.Tensor
                   Output tensor with shape `[batch, dims, seq_len]`.
               """

        if len(x.shape) == 2:
            x = F.one_hot(x, num_classes=self.in_symbols.item()).float()

        assert(len(x.shape) == 3)

        x = self.embedding_layer(x)
        x = x.transpose(1, -1)

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Create NEAR embeddings from an input sequence

               Parameters
               ----------
               x : torch.Tensor
                   Input tensor of shape `[batch, alphabet_size, seq_len]` or `[batch, seq_len]`.

               Returns
               -------
               torch.Tensor
                   Output tensor with shape `[batch, dims, seq_len]`.
               """

        x = self.embed(x)
        x = self.conv_layers(x)
        return x
