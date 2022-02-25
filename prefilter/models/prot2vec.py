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
        learning_rate,
        train_files,
        val_files,
        emission_files,
        schedule_lr,
        step_lr_step_size,
        step_lr_decay_factor,
        batch_size,
        num_workers,
        name_to_class_code,
        normalize_output_embedding=True,
        training=True,
        n_emission_sequences=None,
        pos_weight=1,
        distill=False,
        subsample_neg_labels=False,
        xent=False,
        decoy_files=None,
    ):

        super(Prot2Vec, self).__init__(
            learning_rate=learning_rate,
            train_files=train_files,
            val_files=val_files,
            emission_files=emission_files,
            schedule_lr=schedule_lr,
            step_lr_step_size=step_lr_step_size,
            step_lr_decay_factor=step_lr_decay_factor,
            batch_size=batch_size,
            num_workers=num_workers,
            name_to_class_code=name_to_class_code,
            n_emission_sequences=n_emission_sequences,
            distill=distill,
            xent=xent,
            decoy_files=decoy_files,
        )

        self.res_block_n_filters = res_block_n_filters
        self.vocab_size = vocab_size
        self.res_block_kernel_size = res_block_kernel_size
        self.n_res_blocks = n_res_blocks
        self.res_bottleneck_factor = res_bottleneck_factor
        self.dilation_rate = dilation_rate
        self.normalize_output_embedding = normalize_output_embedding
        self.pos_weight = pos_weight
        self.n_classes = len(name_to_class_code)
        self.subsample_neg_labels = subsample_neg_labels

        if self.xent:
            self.loss_func = torch.nn.CrossEntropyLoss()
            self.class_act = torch.nn.Softmax()
        else:
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
        return x.mean(axis=-1)

    def _forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        return x.mean(axis=-1)

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings = self._masked_forward(x, mask)

        if self.normalize_output_embedding:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)

        classified = self.classification_layer(embeddings)

        return classified

    def _shared_step(self, batch):

        features, masks, labels = batch
        logits = self.forward(features, masks)

        if not self.distill:
            logits = logits.ravel()
            labels = labels.ravel()
            # torch where returns tuple
            # if we're not distilling, throw away positions that contain labels that are not
            # 0 or 1.
            # I'm thinking that there are a huge amount of 0s still, but maybe enough to .
            # still get a good positive signal.
            bad = torch.where((labels == 0) | (labels == 1))[0]
            logits = logits[~bad]
            labels = labels[~bad]

        loss = self.loss_func(logits, labels.float())

        return loss, logits, labels
