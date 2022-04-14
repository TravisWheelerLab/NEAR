import pdb
from abc import ABC

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

from prefilter.models import ResidualBlock, SupConLoss, SupConPerAA, SupConWithPooling
import prefilter.utils as utils
from pathlib import Path

__all__ = ["ResNet1d"]


class ResNet1d(pl.LightningModule, ABC):
    def __init__(
        self,
        fasta_files,
        valid_files,
        logo_path,
        name_to_class_code,
        learning_rate,
        batch_size,
        oversample_neighborhood_labels,
        num_workers=32,
        training=True,
        emission_files=None,
        decoy_files=None,
        padding="valid",
        max_pool=True,
    ):

        super(ResNet1d, self).__init__()

        self.fasta_files = fasta_files
        self.valid_files = valid_files
        self.logo_path = logo_path
        self.name_to_class_code = name_to_class_code
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.padding = padding

        self.oversample_neighborhood_labels = oversample_neighborhood_labels
        self.emission_files = emission_files
        self.decoy_files = decoy_files
        self.num_workers = num_workers

        self.vocab_size = len(utils.PROT_ALPHABET)
        self.res_block_n_filters = 256
        self.feat_dim = 128
        self.res_block_kernel_size = 3
        self.n_res_blocks = 18
        self.res_bottleneck_factor = 1
        self.max_pool = max_pool

        if self.max_pool:
            self.loss_func = SupConWithPooling()
            self.collate_fn = utils.pad_contrastive_batches_with_labelvecs
        else:
            self.loss_func = SupConPerAA()
            self.collate_fn = utils.pad_contrastive_batches_with_labelvecs

        self._setup_layers()

        if training:
            self._create_datasets()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.initial_conv = nn.Conv1d(
            in_channels=self.vocab_size,
            out_channels=self.res_block_n_filters,
            kernel_size=self.res_block_kernel_size,
            padding=self.padding,
        )

        self.embedding_trunk = torch.nn.ModuleList()
        self.projection = nn.Linear(self.res_block_n_filters, self.feat_dim)

        for layer_index in range(self.n_res_blocks):
            self.embedding_trunk.append(
                ResidualBlock(
                    self.res_block_n_filters,
                    self.res_bottleneck_factor,
                    self.res_block_kernel_size,
                    layer_index=layer_index,
                    padding=self.padding,
                )
            )

    def _masked_forward(self, x, mask):
        """
        Before each convolution or batch normalization operation, zero-out
        the features in any location that is padded in the input sequence
        """
        x = self.initial_conv(x)

        if self.padding == "valid":
            mask = mask[
                :,
                :,
                self.res_block_kernel_size // 2 : -(self.res_block_kernel_size // 2),
            ]

        x[mask.expand(-1, self.res_block_n_filters, -1)] = 0

        for i, layer in enumerate(self.embedding_trunk):
            if self.max_pool and (i + 1) % 9 == 0:
                x = torch.nn.functional.max_pool1d(x, 2)
                # is this what I want?
                # masked positions are 1s in the mask.
                # so the positive masks will be propagated inwards.
                # if effect chopping off bits of the end of the sequence.
                mask = torch.nn.functional.max_pool1d(mask.float(), 2).bool()
            x, mask = layer(x, mask)

        return x, mask

    def _forward(self, x):
        x = self.initial_conv(x)
        for i, layer in enumerate(self.embedding_trunk):
            if self.max_pool and (i + 1) % 9 == 0:
                x = torch.nn.functional.max_pool1d(x, 2)
            x = layer(x)
        return x

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings, mask = self._masked_forward(x, mask)

        embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2)

        if mask is not None:
            # always normalize across embedding dimension
            return embeddings, mask
        else:
            return embeddings

    def _shared_step(self, batch):
        if len(batch) == 4:
            features, masks, labelvecs, labels = batch
        else:
            features, masks, labels = batch

        embeddings, masks = self.forward(features, masks)

        if self.max_pool:
            if self.global_step % 250 == 0:
                loss = self.loss_func(
                    embeddings,
                    self.batch_size,
                    picture_path=self.logger.log_dir,
                    step=self.global_step,
                )
            else:
                loss = self.loss_func(embeddings, self.batch_size)
        else:
            # per-AA loss.
            if self.global_step % 250 == 0:
                loss = self.loss_func(
                    embeddings,
                    masks,
                    labelvecs,
                    self.batch_size,
                    picture_path=self.logger.log_dir,
                    step=self.global_step,
                )
            else:
                loss = self.loss_func(embeddings, masks, labelvecs, self.batch_size)

        return loss, embeddings, labels

    def _create_datasets(self):
        # This will be shared between every model that I train.
        if self.max_pool:
            self.train_dataset = utils.RealisticAliPairGenerator()
            self.valid_dataset = utils.RealisticAliPairGenerator(steps_per_epoch=1000)
        else:
            self.train_dataset = utils.NonDiagonalAliPairGenerator()
            self.valid_dataset = utils.NonDiagonalAliPairGenerator(steps_per_epoch=1000)

    def training_step(self, batch, batch_nb):
        loss, _, _ = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss, _, _ = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

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

    def train_dataloader(self):

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        return train_loader

    def val_dataloader(self):

        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        return valid_loader


if __name__ == "__main__":
    # tests
    convnet_1d = ResNet1d()

    data_1d = torch.rand((32, len(PROT_ALPHABET), 105))

    print(convnet_1d(data_1d).shape)
