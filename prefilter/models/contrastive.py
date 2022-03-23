from abc import ABC

import torch
import torch.nn as nn
import pytorch_lightning as pl

from prefilter.models import ResidualBlock
import prefilter.utils as utils
from .losses import SupConLoss
from pathlib import Path

__all__ = ["ResNet1d"]


class ResNet1d(pl.LightningModule, ABC):
    def __init__(
        self,
        fasta_files,
        logo_path,
        name_to_class_code,
        learning_rate,
        batch_size,
        num_workers=32,
        training=True,
        emission_files=None,
        decoy_files=None,
    ):

        super(ResNet1d, self).__init__()

        self.fasta_files = fasta_files
        self.logo_path = logo_path
        self.name_to_class_code = name_to_class_code
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.emission_files = emission_files
        self.decoy_files = decoy_files
        self.num_workers = num_workers

        self.vocab_size = len(utils.PROT_ALPHABET)
        self.res_block_n_filters = 256
        self.feat_dim = 128
        self.res_block_kernel_size = 3
        self.n_res_blocks = 18
        self.res_bottleneck_factor = 1

        self.loss_func = SupConLoss()

        self._setup_layers()

        if training:
            self._create_datasets()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.initial_conv = nn.Conv1d(
            in_channels=self.vocab_size,
            out_channels=self.res_block_n_filters,
            kernel_size=self.res_block_kernel_size,
            padding="same",
        )

        self.embedding_trunk = torch.nn.ModuleList()
        self.projection = nn.Linear(self.res_block_n_filters, self.feat_dim)

        for layer_index in range(self.n_res_blocks):
            self.embedding_trunk.append(
                ResidualBlock(
                    self.res_block_n_filters,
                    self.res_bottleneck_factor,
                    self.res_block_kernel_size,
                    layer_index,
                    1,
                    dilation_rate=None,
                )
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
        return self.projection(x.mean(axis=-1))

    def _forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        return self.projection(x.mean(axis=-1))

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings = self._masked_forward(x, mask)

        # always normalize for contrastive learning
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
        return embeddings

    def _shared_step(self, batch):

        features, masks, labels = batch
        embeddings = self.forward(features, masks)

        f1, f2 = torch.split(embeddings, self.batch_size, dim=0)
        embeddings = torch.cat((f1.unsqueeze(1), f2.unsqueeze(1)), dim=1)

        loss = self.loss_func(embeddings, labels.float())

        return loss, embeddings, labels

    def _create_datasets(self):
        # This will be shared between every model that I train.
        if self.emission_files is not None:
            self.fasta_files = self.emission_files + self.fasta_files

        if self.decoy_files is not None:
            self.fasta_files = self.decoy_files + self.fasta_files

        self.train_dataset = utils.ContrastiveGenerator(
            self.fasta_files,
            self.logo_path,
            self.name_to_class_code,
        )
        # how do i benchmark? Just loss, I guess.
        # hmmm. look at code to do this in published repos...
        self.n_classes = len(self.name_to_class_code)

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
        self.trainer.save_checkpoint(
            Path(self.trainer.checkpoint_callback.dirpath)
            / f"ckpt-{self.global_step}-{loss.item()}.ckpt"
        )

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        self.log("val_loss", val_loss)

    def train_dataloader(self):

        collate_fn = utils.pad_view_batches

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

        return train_loader


if __name__ == "__main__":
    # tests
    convnet_1d = ResNet1d()

    data_1d = torch.rand((32, len(PROT_ALPHABET), 105))

    print(convnet_1d(data_1d).shape)
