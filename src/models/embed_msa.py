import os
import pdb

import esm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.models import ModelBase
from src.utils.layers import PositionalEncoding, ResConv
from src.utils.losses import SupConLoss


class MSAEmbedder(ModelBase):
    def __init__(self, learning_rate, log_interval, training=True):

        super(MSAEmbedder, self).__init__()

        self.learning_rate = learning_rate
        self.training = training

        self.msa_transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_transformer.eval()
        self.msa_transformer.requires_grad_ = False

        self.res_block_n_filters = 768
        self.msa_mlp = torch.nn.Sequential(
            *[torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1)]
        )
        self.seq_mlp = torch.nn.Sequential(
            *[torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1)]
        )

        self.res_block_kernel_size = 5
        self.n_res_blocks = 18
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "circular"

        self.log_interval = log_interval

        self.loss_func = SupConLoss()

        self._setup_layers()

        self.save_hyperparameters()

    def collate_fn(self):
        return None

    def _setup_layers(self):

        self.embed = nn.Embedding(27, self.res_block_n_filters)

        _list = []
        for _ in range(self.n_res_blocks):
            _list.append(
                ResConv(
                    self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                )
            )

        self.embedding_trunk = torch.nn.Sequential(*_list)

        mlp_list = [
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
        ]
        self.mlp = torch.nn.Sequential(*mlp_list)

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x.transpose(-1, -2))
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
        if len(batch) == 3:
            msas, seqs, labels = batch
        else:
            msas, msa_labels, seqs, seq_labels, labels = batch

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
                fpath = (f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",)
                self.logger.experiment.add_figure(
                    f"image_{self.global_step}", plt.gcf()
                )

        _msa = torch.cat(torch.unbind(msa_embeddings, dim=0))
        _seq = torch.cat(torch.unbind(sequence_embeddings, dim=0))
        _msa = torch.nn.functional.normalize(_msa, dim=-1)
        _seq = torch.nn.functional.normalize(_seq, dim=-1)
        # now, drop ye old masked characters.
        x = torch.cat((_msa.unsqueeze(1), _seq.unsqueeze(1)), dim=1)
        loss = self.loss_func(x)

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
