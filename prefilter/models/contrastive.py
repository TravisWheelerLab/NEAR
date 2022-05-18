import pdb
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
    def __init__(
        self,
        learning_rate,
        apply_mlp,
        use_embedding_layer_from_transformer=False,
        training=True,
    ):

        super(ResNet1d, self).__init__()

        self.learning_rate = learning_rate
        self.apply_mlp = apply_mlp
        self.use_embedding_layer_from_transformer = use_embedding_layer_from_transformer
        self.training = training

        self.res_block_n_filters = 1280

        self.res_block_kernel_size = 3
        self.n_res_blocks = 5
        self.res_bottleneck_factor = 1
        self.padding = "same"

        self.log_interval = 100

        self.loss_func = model_utils.SupConLoss()

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        if self.use_embedding_layer_from_transformer:
            # load transformer and grab the embedding layer.
            model, _ = esm.pretrained.esm1b_t33_650M_UR50S()
            self.embed = model.embed_tokens

        _list = []
        for _ in range(self.n_res_blocks):
            _list.append(
                model_utils.ResConv(
                    self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                )
            )
        self.embedding_trunk = _list

        if self.apply_mlp:
            mlp_list = [
                # double up the number of input channels b/c of bidirectional LSTM.
                torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
            ]
            self.mlp = torch.nn.Sequential(*mlp_list)

    def _forward(self, x):
        x = self.embed(x)
        # removed a transpose for the LSTM.
        x = self.embedding_trunk(x.transpose(-1, -2))
        if self.apply_mlp:
            x = self.mlp(x)
        return x

    def _masked_forward(self, x, mask):
        x = self.embed(x).transpose(-1, -2)
        x = ~mask[:, None, :] * x
        for layer in self.embedding_trunk:
            x, mask = layer.masked_forward(x, mask)
        # mask here
        # removed a transpose for the LSTM.
        if self.apply_mlp:
            # point-wise mlp; no downsampling.
            # no need t mask the mask.
            x = self.mlp(x)
            x = ~mask[:, None, :] * x
        return x, mask

    def forward(self, x, masks=None):
        if masks is not None:
            embeddings, masks = self._masked_forward(x, masks)
        else:
            embeddings = self._forward(x)
        return embeddings, masks

    def _shared_step(self, batch):
        if len(batch) == 4:
            features, masks, labelvecs, labels = batch
        else:
            features, masks, labelvecs = batch

        if masks is not None:
            embeddings, masks = self.forward(features, masks)
        else:
            embeddings = self.forward(features)

        # now, do max pooling:

        embeddings, _ = torch.max(embeddings, dim=-1)
        e1, e2 = torch.split(embeddings, embeddings.shape[0] // 2, dim=0)
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
