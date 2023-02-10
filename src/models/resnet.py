from typing import IO, Callable, Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.models.mean_pool import ResidualBlock


class ResNetParamFactory(pl.LightningModule):
    def __init__(
        self,
        n_res_blocks=10,
        res_block_n_filters=128,
    ):

        super(ResNetParamFactory, self).__init__()

        self.initial_conv = nn.Conv1d(
            in_channels=20,
            out_channels=res_block_n_filters,
            kernel_size=3,
            padding="same",
        )

        self.embedding_trunk = []

        for layer_index in range(n_res_blocks):
            self.embedding_trunk.append(
                ResidualBlock(
                    res_block_n_filters,
                    1,
                    3,
                    layer_index,
                    1,
                    dilation_rate=None,
                )
            )
        self.embedding_trunk = torch.nn.Sequential(*self.embedding_trunk)

    def load_from_checkpoint(
        self,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        return self

    def forward(self, x):
        return self.embedding_trunk(self.initial_conv(x)).mean(dim=-1)

    def _shared_step(self, batch):
        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        return optim


class ResNet10M(ResNetParamFactory):
    # 12M params here
    # so 12/18 = 0.66M params/block
    def __init__(self, n_res_blocks=18, res_block_n_filters=128):
        super(ResNet10M, self).__init__(
            n_res_blocks=n_res_blocks, res_block_n_filters=res_block_n_filters
        )


class ResNet50M(ResNetParamFactory):
    def __init__(self, n_res_blocks=750, res_block_n_filters=128):
        super(ResNet50M, self).__init__(
            n_res_blocks=n_res_blocks, res_block_n_filters=res_block_n_filters
        )


class ResNet100M(ResNetParamFactory):
    def __init__(self, n_res_blocks=1500, res_block_n_filters=128):
        super(ResNet100M, self).__init__(
            n_res_blocks=n_res_blocks, res_block_n_filters=res_block_n_filters
        )
