import pdb

import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn

from prefilter import MASK_FLAG
from prefilter.models.base_model import BaseModel
from prefilter.models.layers import ResidualBlock, MultiReceptiveFieldBlock

__all__ = ["Prot2Vec"]


class Prot2Vec(BaseModel):
    """
    Convolutional network for protein family prediction.
    """

    def __init__(
        self,
        res_block_n_filters,
        vocab_size,
        res_block_kernel_size,
        n_res_blocks,
        res_bottleneck_factor,
        dilation_rate,
        normalize_output_embedding=True,
        training=True,
        fcnn=False,
        pos_weight=1,
        **kwargs,
    ):

        super(Prot2Vec, self).__init__(**kwargs)

        self.res_block_n_filters = res_block_n_filters
        self.vocab_size = vocab_size
        self.res_block_kernel_size = res_block_kernel_size
        self.n_res_blocks = n_res_blocks
        self.res_bottleneck_factor = res_bottleneck_factor
        self.dilation_rate = dilation_rate
        self.normalize_output_embedding = normalize_output_embedding
        self.fcnn = fcnn
        self.pos_weight = pos_weight

        self.loss_func = torch.nn.BCEWithLogitsLoss(torch.tensor(self.pos_weight))
        self.class_act = torch.nn.Sigmoid()

        if training:
            self._create_datasets()
            self._init_metrics()

        self._setup_layers()

        if training:
            # TODO: Why is self.hparams not intialized before self.save_hyperparameters() is called?
            self.save_hyperparameters()
            self.hparams["name_to_class_code"] = self.name_to_class_code
            self.hparams["n_classes"] = self.n_classes
            self.save_hyperparameters()

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
                    self.dilation_rate,
                )
            )

        if self.fcnn:
            self.multi_receptive = MultiReceptiveFieldBlock(
                self.res_block_n_filters, self.res_block_n_filters // 10
            )
            self.classification_layer = torch.nn.Conv1d(
                self.res_block_n_filters // 10, self.n_classes, kernel_size=(1,)
            )
        else:
            self.classification_layer = torch.nn.Linear(
                self.res_block_n_filters, self.n_classes
            )

    def _masked_forward(self, x, mask):
        """
        Before each convolution or batch normalization operation, we zero-out
        the features in any location is padded in the input
        sequence
        """
        x = self.initial_conv(x)

        for layer in self.embedding_trunk:
            x = layer(x, mask)
        # re-zero regions
        x[mask.expand(-1, self.res_block_n_filters, -1)] = 0
        # and do an aggregation operation
        # TODO: replace denominator of mean with the correct
        # sequence length. Also add two learnable params:
        # a power on the denominator and numerator
        if self.fcnn:
            return x
        else:
            return x.mean(axis=-1)

    def _forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        if self.fcnn:
            return x
        else:
            return x.mean(axis=-1)

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings = self._masked_forward(x, mask)

        if self.normalize_output_embedding:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)

        if self.fcnn:
            embeddings = self.multi_receptive(embeddings)
            classified = self.classification_layer(embeddings)
        else:
            classified = self.classification_layer(embeddings)

        return classified

    def _shared_step(self, batch):

        features, masks, labels = batch
        labels = labels.int()
        logits = self.forward(features, masks)
        preds = torch.round(self.class_act(logits))

        if self.fcnn:
            labels = labels.unsqueeze(-1)
            labels = labels.expand(-1, -1, logits.shape[-1])

        loss = self.loss_func(logits, labels.float())
        acc = self.accuracy(preds, labels)

        return loss, acc, logits, labels
