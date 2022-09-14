import logging
import os
import pdb

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.models.dot_prod_model import ResNet
from src.utils.layers import PositionalEncoding, ResConv
from src.utils.losses import SupConLoss

logger = logging.getLogger("train")


class SequenceVAE(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        log_interval,
        cnn_model_state_dict,
        initial_seq_len,
        training=True,
    ):

        super(SequenceVAE, self).__init__()

        self.learning_rate = learning_rate
        self.training = training
        self.initial_seq_len = initial_seq_len

        self.cnn_model_state_dict = cnn_model_state_dict

        cnn_model_args = {
            "emb_dim": 256,
            "blocks": 5,
            "block_layers": 2,
            "first_kernel": 11,
            "kernel_size": 5,
            "groups": 2,
            "padding_mode": "reflect",
        }

        self.cnn_model = ResNet(**cnn_model_args)

        success = self.cnn_model.load_state_dict(
            torch.load(cnn_model_state_dict, map_location=torch.device(self.device))
        )
        logger.info(f"{success} for {self.cnn_model_state_dict}")

        self.cnn_model.eval()
        # now freeze it:
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        self.res_block_kernel_size = 3
        self.res_block_n_filters = 256
        # small number of residual blocks
        self.n_res_blocks = 12
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "reflect"
        self.log_interval = log_interval
        self.loss_func = SupConLoss()
        self.one_hot_dimension = 20
        self.layer_list = []

        self._setup_layers()
        self.to("cuda")
        self.save_hyperparameters()
        self.KLD = 0
        self.xent = torch.nn.CrossEntropyLoss()

    def _setup_layers(self):

        self.embed = nn.Conv1d(
            in_channels=self.one_hot_dimension,
            out_channels=self.res_block_n_filters,
            kernel_size=1,
        )

        self.layer_list.append(self.embed)
        z = ResConv(
            self.res_block_n_filters,
            kernel_size=self.res_block_kernel_size,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )

        self.layer_list.append(z)
        self.layer_list.append(torch.nn.AvgPool1d(kernel_size=2))
        self.layer_list.append(
            ResConv(
                self.res_block_n_filters,
                kernel_size=self.res_block_kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode,
            )
        )
        sigma_mlp_list = [
            torch.nn.Linear(
                self.res_block_n_filters * self.initial_seq_len // 2,
                self.res_block_n_filters * self.initial_seq_len // 2,
            ),
        ]
        mu_mlp_list = [
            torch.nn.Linear(
                self.res_block_n_filters * self.initial_seq_len // 2,
                self.res_block_n_filters * self.initial_seq_len // 2,
            ),
        ]

        self.sigma_mlp = torch.nn.Sequential(*sigma_mlp_list)
        self.mu_mlp = torch.nn.Sequential(*mu_mlp_list)

        upsample = torch.nn.Upsample(scale_factor=2)

        # more complexity on the upsampling head.
        upsample_list = [
            ResConv(
                self.res_block_n_filters,
                kernel_size=self.res_block_kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            upsample,
            ResConv(
                self.res_block_n_filters,
                kernel_size=self.res_block_kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            ResConv(
                self.res_block_n_filters,
                kernel_size=self.res_block_kernel_size,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            torch.nn.Conv1d(
                in_channels=self.res_block_n_filters,
                out_channels=self.one_hot_dimension,
                kernel_size=1,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
        ]
        self.upsampler = torch.nn.Sequential(*upsample_list)
        self.final_conv = torch.nn.Conv1d(
            in_channels=self.one_hot_dimension,
            out_channels=self.one_hot_dimension,
            kernel_size=1,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )

    def forward(self, x):
        for layer in self.layer_list:
            layer = layer.to("cuda")
            x = layer(x)
        # flatten the embeddings,
        # pass them into an MLP.
        sigma = self.sigma_mlp(x.reshape(-1, x.shape[-1] * x.shape[-2]))
        mu = self.mu_mlp(x.reshape(-1, x.shape[-1] * x.shape[-2]))
        sample = self.sample(sigma, mu).reshape(-1, x.shape[1], x.shape[2])
        reconstruct = self.final_conv(self.upsampler(sample))
        return sample, reconstruct

    def sample(self, sigma, mu):
        sigma = torch.exp(0.5 * sigma)
        eps = torch.randn_like(sigma)
        self.KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return (eps * sigma) + mu

    def _shared_step(self, batch):
        features, masks, labelvecs = batch
        sampled, recon = self.forward(features)
        # reconstruction loss
        loss = self.xent(recon, features)
        loss += self.KLD

        if self.global_step % self.log_interval == 0:

            e1 = torch.cat(torch.unbind(recon, dim=0))
            e2 = torch.cat(torch.unbind(features, dim=0))
            with torch.no_grad():
                fig, ax = plt.subplots(ncols=2)
                ax[0].imshow(
                    torch.nn.functional.softmax(e1, dim=1)
                    .to("cpu")
                    .numpy()
                    .astype(float),
                    interpolation="nearest",
                )
                ax[1].imshow(
                    e2.to("cpu").numpy().astype(float), interpolation="nearest"
                )
                self.logger.experiment.add_figure(
                    f"image", plt.gcf(), global_step=self.global_step
                )

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
