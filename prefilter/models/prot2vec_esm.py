import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import esm

import utils.datasets as datasets
from utils.utils import (PROT_ALPHABET, pad_word2vec_batch,
        pad_word2vec_batch_with_string)

def esm_embedding(protein_strings):
    batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_strings)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    return token_representations

__all__ = ['Prot2VecESM', 'PROT2VEC_CONFIG']

PROT2VEC_CONFIG = {
        'dilation_rate':3,
        'initial_dilation_rate':2,
        'n_filters': 512,
        'vocab_size':len(PROT_ALPHABET),
        'pooling_layer_type':'avg',
        'kernel_size':3,
        'n_res_blocks':10,
        'bottleneck_factor':0.5,
        'embedding_dim':128
        }

class ResidualBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, 
            bottleneck_factor, kernel_size, stride=1):

        super(ResidualBlock, self).__init__()

        out_channels = int(out_channels*bottleneck_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=1)

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=1)

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.up_bottleneck = nn.Conv1d(out_channels, 
                int(out_channels/bottleneck_factor),
                kernel_size=1, stride=1, padding=0)

    def _masked_forward(self, x, mask):

        x[mask.expand(-1, self.in_channels, -1)] = 0

        out = self.conv1(x)
        out[mask.expand(-1, self.out_channels, -1)] = 0
        out = F.relu(self.bn1(out))

        out[mask.expand(-1, self.out_channels, -1)] = 0
        out = self.conv2(out)

        out[mask.expand(-1, self.out_channels, -1)] = 0
        out = F.relu(self.bn2(out))

        out[mask.expand(-1, self.out_channels, -1)] = 0
        out = self.up_bottleneck(out)

        return out + x

    def _forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.up_bottleneck(out)
        return out + x

    def forward(self, x, mask=None):
        if mask is None:
            return self._forward(x)
        else:
            return self._masked_forward(x, mask)


class Prot2VecESM(pl.LightningModule):

    """ 
    Convolutional network for protein family prediction.

    """

    def __init__(self, args, evaluating=False):

        super().__init__()

        for k, v in args.items():
            setattr(self, k, v)

        self.trunk, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()

        if not evaluating:
            self._create_datasets()

        self.vocab_size = args['vocab_size']

        self._setup_layers()

        self.save_hyperparameters()


    def _setup_layers(self):

        self.initial_conv = nn.Conv1d(in_channels=self.vocab_size,
                                     out_channels=self.n_filters,
                                     kernel_size=self.kernel_size,
                                     padding=1)

        self.bn1 = nn.BatchNorm1d(self.n_filters)

        self.dilated1 = nn.Conv1d(in_channels=self.n_filters,
                out_channels=int(self.n_filters*self.bottleneck_factor),
                kernel_size=self.kernel_size,
                padding=1, stride=1)

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
                              self.bottleneck_factor,
                              self.kernel_size)

            self.encoding_network.append(r)

        if self.pooling_layer_type == 'max':
            self.pool = nn.MaxPool1d()
        if self.pooling_layer_type == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        else:
            raise ValueError('pool type must be one of <max,avg>')

        self.embedding = nn.Linear(self.n_filters, self.embedding_dim)
        self.class_act = nn.Sigmoid()


    def _masked_forward(self, x, mask):
        """
        Before each convolution or batch normalization operation, we zero-out
        the features in any location that corresponds to padding in the input
        sequence 
        """

        out = self.initial_conv(x)
        out[mask.expand(-1, self.n_filters, -1)] = 0
        x = F.relu(self.bn1(out))
        x[mask.expand(-1, self.n_filters, -1)] = 0
        x = self.dilated1(x)
        x[mask.expand(-1, int(self.n_filters*self.bottleneck_factor), -1)] = 0
        x = F.relu(self.bn2(x))
        x[mask.expand(-1, int(self.n_filters*self.bottleneck_factor), -1)] = 0
        x = self.bottleneck1(x) + out
        for layer in self.encoding_network:
            x = layer(x, mask) # takes care of masking in the function
        x = self.pool(x)
        x = self.embedding(x.squeeze())

        return x

    def _forward(self, x):
        out = self.initial_conv(x)

        x = F.relu(self.bn1(out))
        x = self.dilated1(x)
        x = F.relu(self.bn2(x))
        x = self.bottleneck1(x) + out

        for layer in self.encoding_network:
            x = layer(x)

        x = self.pool(x)
        x = self.embedding(x.squeeze())
        return x
    
    def _esm_embedding(self, protein_strings):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(protein_strings)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        return token_representations


    def forward(self, x, mask=None):
        """ 
        Forward a batch of sequences through network.

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
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings = self._masked_forward(x, mask)

        if self.normalize:
            return torch.nn.functional.normalize(embeddings, dim=-1, p=2)
        else:
            return embeddings

    def _get_dots(self, targets, targets_mask, 
            in_context, in_context_mask,
            out_of_context,
            out_of_context_mask):

        targets_embed = self.forward(targets, targets_mask)

        context_embed = self.forward(in_context, in_context_mask)
        negatives_embed = self.forward(out_of_context, out_of_context_mask) 

        negatives_embed = torch.reshape(negatives_embed, (self.batch_size,
            self.n_negative_samples, self.embedding_dim))


        pos_dots = (targets_embed*context_embed).sum(axis=1).squeeze() 
        neg_dots = torch.bmm(negatives_embed, targets_embed.unsqueeze(2)).squeeze()
        return pos_dots, neg_dots.ravel()

    def _compute_loss_and_preds(self, batch):

        targets, targets_mask, contexts, contexts_mask, negatives,\
                negatives_mask, labels = batch[:7]

        pos_dots, neg_dots = self._get_dots(targets, targets_mask,
                contexts, contexts_mask,
                negatives, negatives_mask)

        logits = torch.cat((pos_dots, neg_dots), axis=0)
        loss = self.loss_func(logits, labels.ravel()) 
        preds = self.class_act(logits).ravel()

        return loss, preds, labels, logits, pos_dots, neg_dots

    def training_step(self, batch, batch_idx):
        y = []
        negative_sequences = batch[-1]
        for i, x in enumerate(negative_sequences):
            key = 'protein{}'.format(i)
            print(key)
            y.append((key, x))
        import pdb
        pdb.set_trace()
        xx = self._esm_embedding(y)
        print(xx)

        exit()
        loss, preds, labels, logits, pos_dots, neg_dots\
                = self._compute_loss_and_preds(batch)
        self.log('loss', loss.item())
        self.log('accuracy', torch.sum(torch.round(preds.ravel()) ==
            labels.ravel())/torch.numel(preds))

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds, labels, logits, pos_dots, neg_dots\
                = self._compute_loss_and_preds(batch[:7])
        
        return loss

    def test_step(self, batch, batch_idx):

        loss, preds, labels, logits, pos_dots, neg_dots\
                = self._compute_loss_and_preds(batch)

        return loss

    # values returned by this func are stored over an epoch

    def _create_datasets(self):


        self.test_data = datasets.Word2VecStyleDataset(self.test_files,
                                  self.max_sequence_length,
                                  evaluating=False,
                                  n_negative_samples=self.n_negative_samples,
                                  return_protein_strings=True
                                  )

        self.train_data = datasets.Word2VecStyleDataset(self.train_files,
                                  self.max_sequence_length,
                                  evaluating=False,
                                  n_negative_samples=self.n_negative_samples,
                                  return_protein_strings=True
                                  )

        self.valid_data = datasets.Word2VecStyleDataset(self.valid_files,
                                  self.max_sequence_length,
                                  evaluating=False,
                                  n_negative_samples=self.n_negative_samples,
                                  return_protein_strings=True
                                  )


    def train_dataloader(self):

        return torch.utils.data.DataLoader(self.train_data, 
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=pad_word2vec_batch_with_string)

    def test_dataloader(self):

        return torch.utils.data.DataLoader(self.test_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=pad_word2vec_batch_with_string)

    def val_dataloader(self):

        return torch.utils.data.DataLoader(self.valid_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=pad_word2vec_batch_with_string)

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def configure_optimizers_decay(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        decay_steps = 4000

        def lr_schedule(step):

            x = self.lr * self.gamma ** int(step / decay_steps)
            return x

        mysched = torch.optim.lr_scheduler.LambdaLR(optim, lr_schedule)
        sched = {'scheduler':mysched, 'interval':'step'} 
        return [optim], [sched]

