import pdb
from abc import ABC

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

from prefilter.models import ResidualBlock, SupConLoss, AllVsAllLoss, CustomLoss
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
        all_vs_all_loss=False,
        supcon_loss_per_aa=False,
        non_diag_alignment=False,
        padding="valid",
    ):

        super(ResNet1d, self).__init__()

        self.fasta_files = fasta_files
        self.valid_files = valid_files
        self.logo_path = logo_path
        self.name_to_class_code = name_to_class_code
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.all_vs_all = False
        self.non_diag = non_diag_alignment
        self.supcon = supcon_loss_per_aa
        self.padding = padding

        if sum([self.all_vs_all, self.non_diag, self.supcon]) > 1:
            raise ValueError("Choose one of <non_diag, all_vs_all, supcon>")

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

        if self.non_diag:
            # number of residual convs plus an initial conv.
            self.loss_func = CustomLoss(n_conv_layers=self.n_res_blocks + 1)
        elif all_vs_all_loss:
            self.loss_func = AllVsAllLoss()
            self.all_vs_all = True
        else:
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

        for layer in self.embedding_trunk:
            x, mask = layer(x, mask)

        if self.all_vs_all or self.supcon or self.non_diag:
            return x, mask
        else:
            return self.projection(x.mean(axis=-1))

    def _forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        if self.all_vs_all or self.supcon or self.non_diag:
            return x
        else:
            return self.projection(x.mean(axis=-1))

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings, mask = self._masked_forward(x, mask)
        if self.supcon or self.all_vs_all or self.non_diag:
            # always normalize across embedding dimension
            embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2)
            return embeddings, mask
        else:
            # always normalize for contrastive learning
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
            return embeddings

    def _shared_step(self, batch):

        if self.non_diag:
            # have to propagate labelvecs thru
            features, masks, labelvecs, labels = batch
            embeddings, masks = self.forward(features, masks)
        else:
            features, masks, labels = batch
            embeddings = self.forward(features, masks)

        # sequences, pairs
        f1, f2 = torch.split(embeddings, self.batch_size, dim=0)
        m1, m2 = torch.split(masks, self.batch_size, dim=0)
        embeddings = torch.cat((f1.unsqueeze(1), f2.unsqueeze(1)), dim=1)
        masks = torch.cat((m1.unsqueeze(1), m2.unsqueeze(1)), dim=1)

        if self.all_vs_all:
            m1, m2 = torch.split(masks, self.batch_size, dim=0)
            if self.global_step % 100 == 0:
                loss = self.loss_func(
                    f1, f2, m1, m2, labels, labels, picture=self.global_step
                )
            else:
                loss = self.loss_func(f1, f2, m1, m2, labels, labels)
        elif self.supcon:
            dot_loss = 0
            supcon_loss = 0
            embeddings = embeddings.transpose(-1, -2)
            # grab the two views
            f1, f2 = torch.unbind(embeddings, 1)
            rev_label = torch.flip(labels, [0])
            # reverse along batch dim
            f2 = torch.flip(f2, [0])
            # dot entry i against entry i in f1, f2
            # and remove the places where the labels are the same
            dots = torch.bmm(f1, f2.transpose(-1, -2))[
                torch.not_equal(labels, rev_label)
            ]
            # now add in the sum.
            dot_loss += torch.sum(dots.ravel())
            picture = False
            if self.global_step % 250 == 0:
                picture = True

            for item in embeddings:
                item = item.transpose(0, 1)
                # this is 100x2x256
                if picture:
                    f1, f2 = torch.unbind(item, 1)
                    matmul = torch.matmul(f1, f2.T)
                    plt.imshow(matmul.detach().cpu().numpy())
                    plt.colorbar()
                    plt.savefig(f"supcon{self.global_step}", bbox_inches="tight")
                    plt.close()
                    _, f2 = torch.unbind(embeddings[1].transpose(0, 1), 1)
                    matmul = torch.matmul(f1, f2.T)
                    plt.imshow(matmul.detach().cpu().numpy())
                    plt.colorbar()
                    plt.savefig(f"supcon_neg{self.global_step}", bbox_inches="tight")
                    plt.close()
                    picture = False

                labels = torch.arange(item.shape[0])
                supcon_loss += self.loss_func(item, labels.float())

            loss = (0.1 * dot_loss) + supcon_loss
        elif self.non_diag:
            if self.global_step % 100 == 0:
                loss = self.loss_func(
                    embeddings, masks, labelvecs, picture=self.global_step
                )
            else:
                loss = self.loss_func(embeddings, masks, labelvecs)
        else:
            loss = self.loss_func(embeddings, labels.float())

        return loss, embeddings, labels

    def _create_datasets(self):
        # This will be shared between every model that I train.
        if self.all_vs_all or self.supcon:
            self.train_dataset = utils.AliPairGenerator()
            self.valid_dataset = utils.AliPairGenerator()
        elif self.non_diag:
            self.train_dataset = utils.NonDiagonalAliPairGenerator()
            self.valid_dataset = utils.NonDiagonalAliPairGenerator()
        else:
            if self.emission_files is not None:
                self.fasta_files = self.emission_files + self.fasta_files

            if self.decoy_files is not None:
                self.fasta_files = self.decoy_files + self.fasta_files

            self.train_dataset = utils.ContrastiveGenerator(
                self.fasta_files,
                self.logo_path,
                self.name_to_class_code,
                self.oversample_neighborhood_labels,
            )

            self.valid_dataset = utils.ContrastiveGenerator(
                self.valid_files,
                self.logo_path,
                self.name_to_class_code,
                oversample_neighborhood_labels=False,
            )

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

        if self.non_diag:
            collate_fn = utils.pad_contrastive_batches_with_labelvecs
        else:
            collate_fn = utils.pad_contrastive_batches

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

        return train_loader

    def val_dataloader(self):
        if self.non_diag:
            collate_fn = utils.pad_contrastive_batches_with_labelvecs
        else:
            collate_fn = utils.pad_contrastive_batches

        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )

        return valid_loader


if __name__ == "__main__":
    # tests
    convnet_1d = ResNet1d()

    data_1d = torch.rand((32, len(PROT_ALPHABET), 105))

    print(convnet_1d(data_1d).shape)