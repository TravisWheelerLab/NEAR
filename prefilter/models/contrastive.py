import pdb
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
    def __init__(self, learning_rate, mlm_task, apply_mlp, training=True):

        super(ResNet1d, self).__init__()

        self.learning_rate = learning_rate
        self.apply_mlp = apply_mlp
        self.mlm_task = mlm_task
        self.training = training

        self.res_block_n_filters = 1024
        self.feat_dim = 128
        self.res_block_kernel_size = 3
        self.n_res_blocks = 18
        self.res_bottleneck_factor = 1
        self.padding = "same"

        self.log_interval = 100

        if self.mlm_task:
            self.loss_func = torch.nn.CrossEntropyLoss()
        self.collate_fn = utils.pad_contrastive_batches_with_labelvecs

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        if self.mlm_task:
            self.embed = nn.Embedding(22, self.res_block_n_filters)
        else:
            self.embed = nn.Embedding(21, self.res_block_n_filters)

        _list = []
        for _ in range(self.n_res_blocks):
            _list.append(
                model_utils.ResConv(
                    self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                )
            )
        _list.append(
            nn.Conv1d(
                in_channels=self.res_block_n_filters,
                out_channels=self.res_block_n_filters,
                kernel_size=self.res_block_kernel_size,
                padding=self.padding,
            )
        )

        self.embedding_trunk = torch.nn.Sequential(*_list)

        if self.apply_mlp:
            mlp_list = [
                torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
            ]
            self.mlp = torch.nn.Sequential(*mlp_list)

        if self.mlm_task:
            # 1x1 conv for classification.
            self.classifier = torch.nn.Conv1d(
                self.res_block_n_filters, len(utils.amino_alphabet), 1
            )

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x.transpose(-1, -2))
        if self.apply_mlp:
            x = self.mlp(x)
        if self.mlm_task and self.training:
            # final conv.
            x = torch.nn.ReLU()(x)
            x = self.classifier(x)
        return x

    def forward(self, x):
        embeddings = self._forward(x)
        return embeddings

    def _shared_step(self, batch):
        if self.mlm_task:
            features, labelvecs, _ = batch
            masks = None
        else:
            if len(batch) == 4:
                features, masks, labelvecs, labels = batch
            else:
                features, masks, labelvecs = batch

        if masks is not None:
            embeddings, masks = self.forward(features, masks)
        else:
            embeddings = self.forward(features)

        if self.mlm_task:
            loss = self.loss_func(embeddings, labelvecs)
        else:
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

        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
