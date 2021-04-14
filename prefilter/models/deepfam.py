"""
Author: Lukas Gosch, Roman Feldbauer
"""
# SPDX-License-Identifier: BSD-3-Clause
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from utils import utils as u


__all__ = ['DeepFam', 'DEEPFAM_CONFIG']

DEEPFAM_CONFIG = {
        'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
        'n_filters': 150,
        'dropout': 0.3,
        'vocab_size': 23,
        'hidden_units': 2000,
        }


class DeepFam(nn.Module):
    """ Convolutional network for protein family prediction.

    PyTorch lightning implementation of DeepFam architecture (original: TensorFlow).

    Parameters
    ----------
    model_dict : dict
        Dictionary storing the hyperparameters and learned parameters of
        the model.
    """

    def __init__(self, model_dict):
        super().__init__()

        self.n_classes = model_dict['n_classes']
        self.vocab_size = model_dict['vocab_size']
        self.kernel_sizes = model_dict['kernel_size']
        self.n_filters = model_dict['n_filters']
        self.dropout = model_dict['dropout']
        self.hidden_units = model_dict['hidden_units']

        # One-Hot-Encoding Layer
        # Convolutional Layers
        for i, kernel in enumerate(self.kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.vocab_size,
                                   out_channels=self.n_filters,
                                   kernel_size=kernel)
            # Initialize Convolution Layer, gain = 1.0 to match tensorflow implementation
            nn.init.xavier_uniform_(conv_layer.weight, gain=1.0)
            conv_layer.bias.data.fill_(0.01)
            self.add_module(f'conv{i + 1}', conv_layer)
            # momentum=1-decay to port from tensorflow
            batch_layer = nn.BatchNorm1d(num_features=self.n_filters,
                                         eps=0.001,
                                         momentum=0.1,
                                         affine=True)
            # tensorflow implementation only updates bias term not gamma
            batch_layer.weight.requires_grad = False
            self.add_module(f'batch{i + 1}', batch_layer)
        self.n_conv_layers = len(self.kernel_sizes)

        # Max-Pooling Layer, yields same output as MaxPooling Layer for sequences of size 1000
        # as used in DeepFam but makes the NN applicable to arbitrary sequence lengths
        self.pooling1 = nn.AdaptiveMaxPool1d(output_size=1)

        self.activation1 = nn.ReLU()

        # Dropout
        self.dropout1 = nn.Dropout(p=self.dropout)

        # Dense NN
        # Hidden Layer
        self.linear1 = nn.Linear(in_features=self.n_filters * len(self.kernel_sizes),
                                 out_features=self.hidden_units)
        # Batch Normalization of Hidden Layer
        self.batch_linear = nn.BatchNorm1d(num_features=self.hidden_units,
                                           eps=0.001,
                                           momentum=0.1,
                                           affine=True)
        self.batch_linear.weight.requires_grad = False
        # Classifcation layer
        self.classification1 = nn.Linear(in_features=self.hidden_units,
                                         out_features=self.n_classes)

        # Initialize linear layers
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.classification1.weight, gain=1.0)
        self.linear1.bias.data.fill_(0.01)
        self.classification1.bias.data.fill_(0.01)


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
        # s.t. sum over filter size is a sum over contiguous memory blocks
        max_pool_layer = []
        for i in range(self.n_conv_layers):
            x_conv = getattr(self, f'conv{i + 1}')(x)
            x_conv = getattr(self, f'batch{i + 1}')(x_conv)
            x_conv = self.activation1(x_conv)
            x_conv = self.pooling1(x_conv)
            max_pool_layer.append(x_conv)
        # Concatenate max_pooling output of different convolutions
        x = torch.cat(max_pool_layer, dim=1)
        x = x.view(-1, x.shape[1])
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.batch_linear(x)
        x = self.activation1(x)
        x = self.classification1(x)
        # no softmax here
        return x
