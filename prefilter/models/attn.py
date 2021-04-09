import numpy as np
import torch
import torch.nn as nn

from data.utils import PROT_ALPHABET

__all__ = ['AttentionModel', 'ATTN_CONFIG']

ATTN_CONFIG = {
        'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
        'vocab_size':len(PROT_ALPHABET),
        'n_filters': 150,
        'dropout': 0.3,
        'pooling_layer_type':'avg',
        'qkv_embed_dim': 16,
        'hidden_units': 200,
        'mha_embed_dim':32,
        'num_heads':2,
        }

class AttentionModel(nn.Module):

    def __init__(self, model_dict):

        super().__init__()

        self.n_classes = model_dict['n_classes']
        self.vocab_size = model_dict['vocab_size']
        self.kernel_sizes = model_dict['kernel_size']
        self.n_filters = model_dict['n_filters']
        self.dropout = model_dict['dropout']
        self.hidden_units = model_dict['hidden_units']
        self.mha_embed_dim = model_dict['mha_embed_dim']
        self.num_heads = model_dict['num_heads']
        self.pooling_layer_type = model_dict['pooling_layer_type']
        self.qkv_embed_dim = model_dict['qkv_embed_dim']

        # One-Hot-Encoding Layer
        # Convolutional Layers
        for i, kernel in enumerate(self.kernel_sizes):
            conv_layer = nn.Conv1d(in_channels=self.vocab_size,
                                   out_channels=self.n_filters,
                                   kernel_size=kernel)
            self.add_module(f'conv{i + 1}', conv_layer)
            # momentum=1-decay to port from tensorflow
            batch_layer = nn.BatchNorm1d(num_features=self.n_filters,
                                         eps=0.001,
                                         momentum=0.1,
                                         affine=True)
            self.add_module(f'batch{i + 1}', batch_layer)

            mha_layer = nn.MultiheadAttention(self.mha_embed_dim,
                                                   self.num_heads)
            self.add_module(f'mha{i + 1}', mha_layer)

        self.n_conv_layers = len(self.kernel_sizes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Max-Pooling Layer, yields same output as MaxPooling Layer for sequences of size 1000
        # as used in DeepFam but makes the NN applicable to arbitrary sequence lengths
        if 'avg' in self.pooling_layer_type:
            self.pool1 = nn.AdaptiveAvgPool1d(output_size=1)
        elif 'max' in self.pooling_layer_type:
            self.pool1 = nn.AdaptiveMaxPool1d(output_size=1)
        else:
            raise ValueError(f'Unknown pooling_layer_type: '
                             f'{pooling_layer_type}')

        self.qkv_embeddings = [nn.Linear(1, self.mha_embed_dim, bias=False).to(self.device),
                               nn.Linear(1, self.mha_embed_dim, bias=False).to(self.device),
                               nn.Linear(1, self.mha_embed_dim, bias=False).to(self.device)]

        self.activation1 = nn.ReLU()

        self.hidden1 = nn.Linear(self.n_filters*self.mha_embed_dim*len(self.kernel_sizes),
                self.hidden_units)

        # Dropout
        self.dropout1 = nn.Dropout(p=self.dropout)

        # Classification layer
        self.classification1 = nn.Linear(in_features=self.hidden_units,
                                         out_features=self.n_classes)



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
            x_pool = self.pool1(x_conv)
            q = self.qkv_embeddings[0](x_pool).permute(1, 0, 2)
            k = self.qkv_embeddings[1](x_pool).permute(1, 0, 2)
            v = self.qkv_embeddings[2](x_pool).permute(1, 0, 2) # for MHA to work.
            x_trans, _ = getattr(self, f'mha{i+1}')(q, k, v, need_weights=False)
            max_pool_layer.append(x_trans.permute(1, 0, 2))

        # Concatenate max_pooling output of different convolutions
        x = torch.cat(max_pool_layer, dim=-1)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = self.hidden1(x) 
        x = self.dropout1(x)
        x = self.classification1(x)
        # no softmax here
        return x
