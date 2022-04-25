import pdb
from abc import ABC

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

from prefilter.models import (
    ResNet,
    SupConLoss,
    SupConPerAA,
    SupConWithPooling,
    SupConNoMasking,
)
import prefilter.utils as utils
from pathlib import Path

__all__ = ["ResNet1d"]


class ResNet1d(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate,
    ):

        super(ResNet1d, self).__init__()

        self.learning_rate = learning_rate

        self.vocab_size = len(utils.PROT_ALPHABET)
        self.res_block_n_filters = 256
        self.feat_dim = 128
        self.res_block_kernel_size = 3
        self.n_res_blocks = 1
        self.res_bottleneck_factor = 1
        self.padding = "valid"

        self.log_interval = 2000

        self.loss_func = SupConNoMasking()
        self.collate_fn = utils.pad_contrastive_batches_with_labelvecs

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.embed = nn.Embedding(21, self.res_block_n_filters)

        _list = []
        for _ in range(3):
            _list.append(
                nn.Conv1d(
                    in_channels=self.res_block_n_filters,
                    out_channels=self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                )
            )
            _list.append(nn.GELU())
        _list.append(nn.MaxPool1d(2))
        for _ in range(2):
            _list.append(
                nn.Conv1d(
                    in_channels=self.res_block_n_filters,
                    out_channels=self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                )
            )
            _list.append(nn.GELU())

        _list.append(
            nn.Conv1d(
                in_channels=self.res_block_n_filters,
                out_channels=self.res_block_n_filters,
                kernel_size=self.res_block_kernel_size,
                padding=self.padding,
            )
        )

        self.embedding_trunk = torch.nn.Sequential(*_list)

    def _masked_forward(self, x, mask):
        """
        Before each convolution or batch normalization operation, zero-out
        the features in any location that is padded in the input sequence
        """
        x = self.embed(x)
        x = self.initial_conv(x)
        mask = mask[:, :, 1:-1]
        mask = utils.mask_mask(mask)
        return x, mask

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x.transpose(-1, -2))
        return x

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings, mask = self._masked_forward(x, mask)

        if mask is not None:
            return embeddings, mask
        else:
            return embeddings

    def _shared_step(self, batch):
        if len(batch) == 4:
            features, masks, labelvecs, labels = batch
        else:
            features, masks, labels = batch

        if masks is not None:
            embeddings, masks = self.forward(features, masks)
        else:
            embeddings = self.forward(features)

        if self.global_step % self.log_interval == 0:
            loss = self.loss_func(
                embeddings,
                masks,
                labelvecs,
                picture_path=self.logger.log_dir,
                step=self.global_step,
            )
        else:
            loss = self.loss_func(embeddings, masks, labelvecs)

        return loss, embeddings, labels

    def training_step(self, batch, batch_nb):
        loss, _, _ = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss, _, _ = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_epoch_end(self, outputs):
        train_loss = self.all_gather([x["loss"] for x in outputs])
        loss = torch.mean(torch.stack(train_loss))
        self.log("train_loss", loss)
        self.log("learning_rate", self.learning_rate)

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        self.log("val_loss", val_loss)
