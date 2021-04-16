import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import PROT_ALPHABET
from .standard import ClassificationTask


__all__ = ['ProtCNN', 'PROTCNN_CONFIG']


PROTCNN_CONFIG = {
        'dilation_rate':3,
        'initial_dilation_rate':2,
        'n_filters': 150,
        'vocab_size':len(PROT_ALPHABET),
        'pooling_layer_type':'avg',
        'kernel_size':21,
        'n_res_blocks':1,
        'bottleneck_factor':0.5,
        }


class ResidualBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, dilation,
            bottleneck_factor, kernel_size, stride=1):

        super(ResidualBlock, self).__init__()

        out_channels = int(out_channels*bottleneck_factor)

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size+9,
            dilation=dilation)

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size+9,
                               dilation=dilation)

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.up_bottleneck = nn.Conv1d(out_channels, int(out_channels/bottleneck_factor),
                kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.up_bottleneck(out)
        return out + x


class ProtCNN(ClassificationTask):
    """ 
    Convolutional network for protein family prediction.

    PyTorch lightning implementation of the Resnet in 
    'Using Deep Learning to Annotate the Protein Universe'
    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict, task_args):

        super().__init__(task_args)

        self.n_classes = model_dict['n_classes']
        self.vocab_size = model_dict['vocab_size']
        self.n_res_blocks = model_dict['n_res_blocks']
        self.initial_dilation_rate = model_dict['initial_dilation_rate']
        self.dilation_rate = model_dict['dilation_rate']
        self.n_filters = model_dict['n_filters']
        self.bottleneck_factor = model_dict['bottleneck_factor']
        self.pool_type = model_dict['pooling_layer_type']
        self.kernel_size = model_dict['kernel_size']

        self.initial_conv = nn.Conv1d(in_channels=self.vocab_size,
                                     out_channels=self.n_filters,
                                     kernel_size=self.kernel_size,
                                     padding=self.kernel_size-1,
                                     dilation=self.initial_dilation_rate)
        # model:
        # one-hot (Lx20)
        # initial conv (LxF) (no dilation (?)) (or is it 2 dilation?)
        # resblock (LxF) (
        # bottleneck: (LxF) 
        # resblock (LxF//2)
        # bottleneck (LxF)
          
        # bottleneck: down, larger kernel, up. A larger kernel size is sandwiched between two 
        # bottleneck layers

        self.bn1 = nn.BatchNorm1d(self.n_filters)

        self.dilated1 = nn.Conv1d(in_channels=self.n_filters,
                out_channels=int(self.n_filters*self.bottleneck_factor),
                kernel_size=self.kernel_size,
                dilation=self.dilation_rate,
                padding=self.kernel_size+9,
                stride=1)

        self.bn2 = nn.BatchNorm1d(int(self.n_filters*self.bottleneck_factor))
        self.bottleneck1 = nn.Conv1d(in_channels=int(self.n_filters*self.bottleneck_factor),
                out_channels=self.n_filters,
                kernel_size=1,
                stride=1,
                padding=0)

        self.encoding_network = nn.ModuleList([])

        for _ in range(self.n_res_blocks):

            r = ResidualBlock(self.n_filters,
                              self.n_filters,
                              self.dilation_rate,
                              self.bottleneck_factor,
                              self.kernel_size).to('cuda')

            self.encoding_network.append(r)

        if self.pool_type == 'max':
            self.pool = nn.MaxPool1d()
        if self.pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        else:
            raise ValueError('pool type must be one of <max,avg>')

        self.classification = nn.Linear(self.n_filters, self.n_classes)

    def forward(self, x):
        """ Forward a batch of sequences through network.

        Parameters
        ----------
        x : Tensor, shape (batch_size, sequence_len)
            Sequence or batch of sequences to classify. Assumes they are
            translated using a vocabulary. (See gen_amino_acid_vocab in
            dataset.py)

        Returns
        -------
        out : Tensor, shape (batch_size, n_classes)
            Confidence of sequence(s) being in one of the n_classes.
        """
        out = self.initial_conv(x)

        x = F.relu(self.bn1(out))
        x = self.dilated1(x)
        x = F.relu(self.bn2(x))
        x = self.bottleneck1(x) + out

        for layer in self.encoding_network:
            x = layer(x)

        x = self.pool(x)
        x = self.classification(x.squeeze())

        return x
