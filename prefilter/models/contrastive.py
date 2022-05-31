import pdb
import os
import esm
from abc import ABC

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

import prefilter.models as model_utils
import prefilter.utils as utils
from pathlib import Path

__all__ = ["ResNet1d"]


class ResNet1d(pl.LightningModule, ABC):
    def __init__(self, learning_rate, embed_msas, apply_attention, training=True):

        super(ResNet1d, self).__init__()

        self.learning_rate = learning_rate
        self.training = training
        self.embed_msas = embed_msas
        self.apply_attention = apply_attention

        if self.embed_msas:
            self.msa_transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            self.msa_transformer.eval()
            self.msa_transformer.requires_grad_ = False

        if self.embed_msas:
            self.res_block_n_filters = 768
            self.msa_mlp = torch.nn.Sequential(
                *[
                    torch.nn.Conv1d(
                        self.res_block_n_filters, self.res_block_n_filters, 1
                    )
                ]
            )
            self.seq_mlp = torch.nn.Sequential(
                *[
                    torch.nn.Conv1d(
                        self.res_block_n_filters, self.res_block_n_filters, 1
                    )
                ]
            )
        else:
            self.res_block_n_filters = 256

        self.res_block_kernel_size = 5
        self.n_res_blocks = 18
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "circular"

        self.log_interval = 100

        self.loss_func = model_utils.SupConLoss()

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.embed = nn.Embedding(27, self.res_block_n_filters)

        _list = []
        for _ in range(self.n_res_blocks):
            _list.append(
                model_utils.ResConv(
                    self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                )
            )

        self.embedding_trunk = torch.nn.Sequential(*_list)
        if self.apply_attention:
            self.transformer = torch.nn.TransformerEncoderLayer(
                self.res_block_n_filters,
                nhead=8,
                dim_feedforward=2 * self.res_block_n_filters,
            )

            self.pos_unc = model_utils.PositionalEncoding(self.res_block_n_filters)
        # could apply attention after max pooling then project back up to original dimension.

        mlp_list = [
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
        ]
        self.mlp = torch.nn.Sequential(*mlp_list)

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x.transpose(-1, -2))
        if self.apply_attention:
            x = x.transpose(1, 0).transpose(0, -1)
            x = self.transformer(x)
            x = x.transpose(1, 0).transpose(-1, -2)
        x = self.mlp(x)
        return x

    def _masked_forward(self, x, mask):
        x = self.embed(x).transpose(-1, -2)
        x = ~mask[:, None, :] * x
        for layer in self.embedding_trunk:
            x, mask = layer.masked_forward(x, mask)
        # mask here
        # removed a transpose for the LSTM.
        # point-wise mlp; no downsampling.
        # no need to mask the mask (kernel size 1).
        x = self.mlp(x)
        x = ~mask[:, None, :] * x
        return x, mask

    def forward(self, x, masks=None):
        if masks is not None:
            embeddings, masks = self._masked_forward(x, masks)
            return embeddings, masks
        else:
            embeddings = self._forward(x)
            return embeddings

    def _shared_step(self, batch):
        if self.embed_msas:
            if len(batch) == 3:
                msas, seqs, labels = batch
            else:
                msas, msa_labels, seqs, seq_labels, labels = batch
        elif len(batch) == 4:
            features, masks, labelvecs, labels = batch
        else:
            features, masks, labelvecs = batch

        if self.embed_msas:
            # each msa gets an _entire_ embedding
            # is there something weird about this?
            msa_embeddings = self.msa_transformer(
                msas, repr_layers=[12], return_contacts=False
            )["representations"][12].detach()
            # remove begin-of-sequence token.
            # msa_embeddings = msa_embeddings[:, :, 1:, :]
            # mean pool sequence embeddings across
            # msa dimension
            # msa_embeddings, _ = torch.max(msa_embeddings, dim=1)
            msa_embeddings = msa_embeddings[:, 0]
            # now apply two mlps.
            sequence_embeddings = (
                self.msa_transformer(seqs, repr_layers=[12], return_contacts=False)[
                    "representations"
                ][12]
                .detach()
                .squeeze()
            )
            # this should work well. If it doesn't something is up.
            sequence_embeddings = sequence_embeddings.transpose(-1, -2)
            msa_embeddings = msa_embeddings.transpose(-1, -2)

            msa_embeddings = self.msa_mlp(sequence_embeddings).transpose(-1, -2)
            sequence_embeddings = self.seq_mlp(sequence_embeddings).transpose(-1, -2)

            if self.global_step % self.log_interval == 0:
                with torch.no_grad():
                    _msa = torch.cat(torch.unbind(msa_embeddings, dim=0))
                    _seq = torch.cat(torch.unbind(sequence_embeddings, dim=0))
                    _msa = torch.nn.functional.normalize(_msa, dim=-1)
                    _seq = torch.nn.functional.normalize(_seq, dim=-1)

                    plt.imshow(torch.matmul(_msa, _seq.T).to("cpu").detach().numpy())
                    plt.colorbar()
                    fpath = (
                        f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",
                    )
                    if os.path.isdir(self.trainer.logger.log_dir):
                        print(f"saving to {fpath}")
                        plt.savefig(
                            f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",
                            bbox_inches="tight",
                        )
                    plt.close()
        else:
            if masks is not None:
                embeddings, masks = self.forward(features, masks)
            else:
                embeddings = self.forward(features)

        if self.embed_msas:
            _msa = torch.cat(torch.unbind(msa_embeddings, dim=0))
            _seq = torch.cat(torch.unbind(sequence_embeddings, dim=0))
            _msa = torch.nn.functional.normalize(_msa, dim=-1)
            _seq = torch.nn.functional.normalize(_seq, dim=-1)
            # now, drop ye old masked characters.
            x = torch.cat((_msa.unsqueeze(1), _seq.unsqueeze(1)), dim=1)
            loss = self.loss_func(x)

        else:
            e1, e2 = torch.split(
                embeddings.transpose(-1, -2), embeddings.shape[0] // 2, dim=0
            )
            e1 = torch.cat(torch.unbind(e1, dim=0))
            e2 = torch.cat(torch.unbind(e2, dim=0))
            e1 = torch.nn.functional.normalize(e1, dim=-1)
            e2 = torch.nn.functional.normalize(e2, dim=-1)
            if self.global_step % self.log_interval == 0:
                with torch.no_grad():
                    plt.imshow(torch.matmul(e1, e2.T).to("cpu").detach().numpy())
                    plt.colorbar()
                    fpath = (
                        f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",
                    )
                    if os.path.isdir(self.trainer.logger.log_dir):
                        print(f"saving to {fpath}")
                        plt.savefig(
                            f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",
                            bbox_inches="tight",
                        )
                    plt.close()
            loss = self.loss_func(torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1))

        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
        # lr_schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=15, gamma=0.5)
        return optim

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
