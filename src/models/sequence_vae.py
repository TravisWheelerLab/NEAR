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
        cnn_model_args,
        pool_factors=[2, 2, 2, 2],
        training=True,
    ):

        super(SequenceVAE, self).__init__()

        self.learning_rate = learning_rate
        self.training = training
        self.pool_factors = pool_factors

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
        self.res_block_n_filters = 128
        # small number of residual blocks
        self.n_res_blocks = 12
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "reflect"
        self.log_interval = log_interval
        self.loss_func = SupConLoss()
        self.one_hot_dimension = 20

        self._setup_layers()
        self.save_hyperparameters()
        self.N = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def _setup_layers(self):

        self.embed = nn.Conv1d(
            in_channels=self.one_hot_dimension,
            out_channels=self.res_block_n_filters,
            kernel_size=1,
        )

        layer_list = [self.embed]

        for i in range(self.n_res_blocks):

            if (i + 1) % 3 == 0:
                layer_list.append(torch.nn.AvgPool1d(self.pool_factors[i // 3]))

            layer_list.append(
                ResConv(
                    self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                )
            )

        self.embedding_trunk = torch.nn.Sequential(*layer_list)

        sigma_mlp_list = [
            torch.nn.Linear(128 * 8, 4096),
            torch.nn.ELU(),
            torch.nn.Linear(4096, 4096),
        ]
        mu_mlp_list = [
            torch.nn.Linear(128 * 8, 4096),
            torch.nn.ELU(),
            torch.nn.Linear(4096, 4096),
        ]

        self.projector = torch.nn.Linear(4096, 128 * 8)

        self.sigma_mlp = torch.nn.Sequential(*sigma_mlp_list)
        self.mu_mlp = torch.nn.Sequential(*mu_mlp_list)

        upsample = torch.nn.Upsample(scale_factor=2)

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
            upsample,
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
            upsample,
            torch.nn.Conv1d(
                in_channels=self.res_block_n_filters,
                out_channels=self.one_hot_dimension,
                kernel_size=1,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
        ]

        self.upsampler = torch.nn.Sequential(*upsample_list)

    def forward(self, x):
        x = self.embedding_trunk(x)
        sigma2 = self.sigma_mlp(x.reshape(-1, x.shape[-1] * x.shape[-2]))
        mu = self.mu_mlp(x.reshape(-1, x.shape[-1] * x.shape[-2]))
        sample = self.reparameterize(sigma2, mu)
        logger.debug("fixme.")
        sample = self.projector(sample)
        # reconstructed = self.upsampler(sample.reshape(-1, x.shape[1], x.shape[2]))
        return sample

    def reparameterize(self, sigma2, mu):
        sigma2 = torch.exp(sigma2)
        z = mu + sigma2 * self.N.sample(mu.shape)
        self.kl = (sigma2**2 + mu**2 - torch.log(sigma2) - 1 / 2).sum()
        return z

    def _shared_step(self, batch):
        features, masks, labelvecs = batch

        (
            sampled,
            reconstructed,
        ) = self.forward(features)

        encoded_features = self.cnn_model(features)
        encoded_reconstructed = self.cnn_model(reconstructed)

        e1 = encoded_features.transpose(-1, -2)
        e2 = encoded_reconstructed.transpose(-1, -2)

        e1 = torch.cat(torch.unbind(e1, dim=0))
        e2 = torch.cat(torch.unbind(e2, dim=0))
        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)

        if self.global_step % self.log_interval == 0:
            # what _should_ the output look like?
            # let's think. What should
            #
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(
                    torch.matmul(e1, e2.T).to("cpu").detach().numpy().astype(float),
                    interpolation="nearest",
                    vmin=-1,
                    vmax=1,
                )
                plt.colorbar()
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
