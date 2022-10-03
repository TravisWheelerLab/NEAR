import logging
import os
import pdb

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.models.dot_prod_model import ResNet
from src.utils.gen_utils import amino_alphabet
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
        downsample_steps,
        pool_type,
        apply_cnn_loss,
        backprop_on_near_aminos,
        apply_contrastive_loss,
        training=True,
    ):

        super(SequenceVAE, self).__init__()

        self.learning_rate = learning_rate
        self.downsample_steps = int(downsample_steps)
        self.training = training
        self.initial_seq_len = int(initial_seq_len)
        self.pool_type = pool_type
        self.apply_cnn_loss = apply_cnn_loss
        self.backprop_on_near_aminos = backprop_on_near_aminos
        self.apply_contrastive_loss = apply_contrastive_loss
        self.cnn_model_state_dict = cnn_model_state_dict
        self.supcon = SupConLoss()

        self.cnn_model_args = {
            "emb_dim": 256,
            "blocks": 5,
            "block_layers": 2,
            "first_kernel": 11,
            "kernel_size": 5,
            "groups": 2,
            "padding_mode": "reflect",
        }

        self.cnn_model = ResNet(**self.cnn_model_args)
        self.cnn_model_state_dict = cnn_model_state_dict

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
        self.pool_type = pool_type

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
            if self.pool_type == "mean":
                self.layer_list.append(torch.nn.AvgPool1d(kernel_size=2))
            else:
                self.layer_list.append(torch.nn.MaxPool1d(kernel_size=2))

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
        # reconstruction loss
        # DISCUSS:
        # does the reconstructed vector match known dirichlet mixtures?
        loss = self.xent(recon, original_features)
        # KLD is quite large.
        loss += self.KLD

        if self.apply_cnn_loss:
            # use the CNN!
            embeds = self.cnn_model(original_features)
            recon_embeds = self.cnn_model(torch.nn.functional.softmax(recon, dim=1))
            # # l2 loss on diag.
            e1 = torch.cat(torch.unbind(embeds, dim=0))
            e2 = torch.cat(torch.unbind(recon_embeds, dim=0))
            dots = torch.cdist(e1, e2)
            # minimize the difference b/t the diagonal elements
            # this _should_ be 0 if the embeddings are the same.
            # not really sure where thhe majority of the loss is coming from
            # is it here?
            if self.backprop_on_near_aminos:
                loss += (torch.diag(dots)[torch.diag(dots) <= 1] ** 2).sum()
            else:
                loss += (torch.diag(dots) ** 2).sum()

        if self.apply_contrastive_loss:
            sampled_mutated, recon_mutated = self.forward(mutated_features)

            recon = torch.cat(torch.unbind(recon, dim=0))
            recon_mutated = torch.cat(torch.unbind(recon_mutated, dim=0))

            recon = torch.nn.functional.normalize(recon, dim=-1)

            recon_mutated = torch.nn.functional.normalize(recon_mutated, dim=-1)

            # fmt: off
            loss += self.supcon(torch.cat((recon_mutated.unsqueeze(1), recon.unsqueeze(1)), dim=1))
            # fmt: on

        if self.global_step % self.log_interval == 0:

            if self.apply_contrastive_loss:
                e1 = recon
                e2 = torch.cat(torch.unbind(original_features, dim=0))
            else:
                e1 = torch.cat(torch.unbind(recon, dim=0))
                e2 = torch.cat(torch.unbind(original_features, dim=0))

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


class SequenceVAEWithIndels(SequenceVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _shared_step(self, batch):
        sequence1, labels1, sequence2, labels2 = batch
        concat_features = torch.cat((sequence1, sequence2), dim=0)
        sampled, recon = self.forward(concat_features)
        loss = self.xent(recon, concat_features)
        loss += self.KLD
        # now comes the weird part.
        # I only want to backpropagate on the aminos that are aligned.
        # this will require masking parts of the loss.
        # this is weird.
        # what if i use a binary cross entropy?

        if self.apply_cnn_loss:
            embeds = self.cnn_model(concat_features)
            recon_embeds = self.cnn_model(torch.nn.functional.softmax(recon, dim=1))
            # recall this is on concatenated features for performance
            embeds1, embeds2 = torch.split(embeds, embeds.shape[0] // 2)
            recon_embeds1, recon_embeds2 = torch.split(
                recon_embeds, embeds.shape[0] // 2
            )
            #
            # i have to split these guys in two again
            # # l2 loss on diag.
            e1 = torch.cat(torch.unbind(embeds1, dim=-1))
            e2 = torch.cat(torch.unbind(recon_embeds2, dim=-1))

            l1 = torch.cat(torch.unbind(labels1, dim=0))
            l2 = torch.cat(torch.unbind(labels2, dim=0))
            labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0))
            # dists[dists > 1] = 0
            dists = torch.cdist(e1, e2)
            loss += dists[labelmat].sum()

        if self.apply_contrastive_loss:
            features1, features2 = torch.split(
                concat_features, concat_features.shape[0] // 2
            )
            recon1, recon2 = torch.split(recon, concat_features.shape[0] // 2)
            e1 = torch.cat(torch.unbind(features1, dim=-1))
            e2 = torch.cat(torch.unbind(recon2, dim=-1))
            # don't normalize e1
            e2 = torch.nn.functional.normalize(e2, dim=0)
            l1 = torch.cat(torch.unbind(labels1, dim=0))
            l2 = torch.cat(torch.unbind(labels2, dim=0))
            labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()
            all_dots = torch.matmul(e1, e2.T)
            loss += torch.nn.functional.binary_cross_entropy_with_logits(
                all_dots.ravel(), labelmat.ravel()
            )

        if self.global_step % self.log_interval == 0:

            e1 = torch.cat(
                torch.unbind(torch.nn.functional.softmax(recon, dim=1), dim=-1)
            )[:200]
            e2 = torch.cat(torch.unbind(concat_features, dim=-1))[:200]

            with torch.no_grad():
                fig, ax = plt.subplots(ncols=2)
                if self.apply_contrastive_loss:
                    acc = (
                        torch.round(torch.sigmoid(all_dots)) == labelmat
                    ).sum() / labelmat.numel()
                    ax[0].set_title(f"accuracy: {acc.item():.5f}")
                ax[0].imshow(
                    e1.to("cpu").numpy().astype(float),
                    interpolation="nearest",
                )
                ax[1].imshow(
                    e2.to("cpu").numpy().astype(float), interpolation="nearest"
                )
                self.logger.experiment.add_figure(
                    f"image", plt.gcf(), global_step=self.global_step
                )

        return loss


class SequenceVAETrainCNN(SequenceVAEWithIndels):
    def __init__(self, pretrained_cnn, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cnn_model = ResNet(**self.cnn_model_args)
        if pretrained_cnn:
            success = self.cnn_model.load_state_dict(
                torch.load(
                    self.cnn_model_state_dict, map_location=torch.device(self.device)
                )
            )
            logger.info(f"{success} for {self.cnn_model_state_dict}")
