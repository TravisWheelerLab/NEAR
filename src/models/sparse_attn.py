import os
import pdb

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.layers import PositionalEncoding, ResConv
from src.utils.losses import SupConLoss


class ResNetSparseAttention(pl.LightningModule):
    def __init__(self, learning_rate, log_interval, training=True):

        super(ResNetSparseAttention, self).__init__()

        self.learning_rate = learning_rate
        self.training = training

        self.res_block_kernel_size = 5
        self.res_block_n_filters = 128
        self.n_res_blocks = 5
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "circular"

        self.log_interval = log_interval

        self.loss_func = SupConLoss()

        self._setup_layers()

        self.save_hyperparameters()

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
        self.transformer = torch.nn.TransformerEncoderLayer(
            self.res_block_n_filters,
            nhead=8,
            dim_feedforward=2 * self.res_block_n_filters,
        )

        self.pos_unc = PositionalEncoding(self.res_block_n_filters)

        mlp_list = [
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
        ]

        self.mlp = torch.nn.Sequential(*mlp_list)

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x.transpose(-1, -2))
        x = x.transpose(1, 0).transpose(0, -1)
        x = self.transformer(x)
        x = x.transpose(1, 0).transpose(-1, -2)
        x = self.mlp(x)
        return x

    def forward(self, x, masks=None):
        if masks is not None:
            embeddings, masks = self._masked_forward(x, masks)
            return embeddings, masks
        else:
            embeddings = self._forward(x)
            return embeddings

    def _shared_step(self, batch):
        features, masks, labelvecs = batch
        embeddings = self.forward(features)

        e1, e2 = torch.split(
            embeddings.transpose(-1, -2), embeddings.shape[0] // 2, dim=0
        )
        e1 = torch.cat(torch.unbind(e1, dim=0))
        e2 = torch.cat(torch.unbind(e2, dim=0))
        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)

        if self.global_step % self.log_interval == 0:

            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(torch.matmul(e1, e2.T).to("cpu").detach().numpy())
                plt.colorbar()
                fpath = (f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",)
                self.logger.experiment.add_figure(
                    f"image", plt.gcf(), global_step=self.global_step
                )

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