import logging
import pdb

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.layers import ResConv
from src.utils.losses import SupConLoss

logger = logging.getLogger("train")


class KMerVAE(pl.LightningModule):
    """
    Reconstruct K-mers.
    """

    def __init__(
        self,
        learning_rate,
        log_interval,
        initial_seq_len,
        downsample_steps,
        training=True,
    ):

        super(KMerVAE, self).__init__()

        self.learning_rate = learning_rate
        self.downsample_steps = int(downsample_steps)
        self.training = training
        self.initial_seq_len = int(initial_seq_len)
        self.supcon = SupConLoss()

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
        self.to(self.device)
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

        for i in range(self.downsample_steps):
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
                self.res_block_n_filters
                * (self.initial_seq_len // 2**self.downsample_steps),
                self.res_block_n_filters
                * (self.initial_seq_len // 2**self.downsample_steps),
            ),
        ]
        mu_mlp_list = [
            torch.nn.Linear(
                self.res_block_n_filters
                * (self.initial_seq_len // 2**self.downsample_steps),
                self.res_block_n_filters
                * (self.initial_seq_len // 2**self.downsample_steps),
            ),
        ]

        self.sigma_mlp = torch.nn.Sequential(*sigma_mlp_list)
        self.mu_mlp = torch.nn.Sequential(*mu_mlp_list)

        upsample = torch.nn.Upsample(scale_factor=2)
        upsample_list = []

        for _ in range(self.downsample_steps):
            upsample_list.extend(
                [
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
                    upsample,
                ]
            )

        upsample_list.append(
            torch.nn.Conv1d(
                in_channels=self.res_block_n_filters,
                out_channels=self.one_hot_dimension,
                kernel_size=1,
                padding=self.padding,
                padding_mode=self.padding_mode,
            )
        )

        self.upsampler = torch.nn.Sequential(*upsample_list)
        self.downsampler = torch.nn.Sequential(*self.layer_list)

        self.final_conv = torch.nn.Conv1d(
            in_channels=self.one_hot_dimension,
            out_channels=self.one_hot_dimension,
            kernel_size=1,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )

    def forward(self, x):
        x = self.downsampler(x)
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
        original_features, mutated_features, _ = batch
        sampled, recon = self.forward(original_features)
        # not going to actually reconstruct the features;
        # going to predict the k-mer at that point.
        loss = self.xent(recon, original_features)
        # KLD is quite large.
        loss += self.KLD
        pdb.set_trace()

        if self.global_step % self.log_interval == 0:

            pass

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
