import math

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.losses import SupConLoss


class ResidualBlock(nn.Module):
    def __init__(
        self,
        filters,
        resnet_bottleneck_factor,
        kernel_size,
        layer_index,
        first_dilated_layer,
        dilation_rate,
        stride=1,
        typ="1d",
    ):

        super(ResidualBlock, self).__init__()

        self.filters = filters

        shifted_layer_index = layer_index - first_dilated_layer + 1
        if dilation_rate is not None:
            dilation_rate = int(max(1, dilation_rate**shifted_layer_index))
        else:
            dilation_rate = 1
        self.num_bottleneck_units = math.floor(resnet_bottleneck_factor * self.filters)
        if typ == "1d":
            self.bn1 = torch.nn.BatchNorm1d(self.filters)
            # need to pad 'same', so output has the same size as input
            # project down to a smaller number of self.filters with a larger kernel size
            self.conv1 = torch.nn.Conv1d(
                self.filters,
                self.num_bottleneck_units,
                kernel_size=kernel_size,
                dilation=dilation_rate,
                padding="same",
            )

            self.bn2 = torch.nn.BatchNorm1d(self.num_bottleneck_units)
            # project back up to a larger number of self.filters w/ a kernel size of 1 (a local
            # linear transformation) No padding needed sin
            self.conv2 = torch.nn.Conv1d(
                self.num_bottleneck_units, self.filters, kernel_size=1, dilation=1
            )
        else:
            self.bn1 = torch.nn.BatchNorm2d(self.filters)
            # need to pad 'same', so output has the same size as input
            # project down to a smaller number of self.filters with a larger kernel size
            self.conv1 = torch.nn.Conv2d(
                self.filters,
                self.num_bottleneck_units,
                kernel_size=kernel_size,
                dilation=dilation_rate,
                padding="same",
            )

            self.bn2 = torch.nn.BatchNorm2d(self.num_bottleneck_units)
            # project back up to a larger number of self.filters w/ a kernel size of 1 (a local
            # linear transformation) No padding needed sin
            self.conv2 = torch.nn.Conv2d(
                self.num_bottleneck_units, self.filters, kernel_size=1, dilation=1
            )

    def _forward(self, x):
        features = self.bn1(x)
        features = torch.nn.functional.relu(features)
        features = self.conv1(features)
        features = self.bn2(features)
        features = torch.nn.functional.relu(features)
        features = self.conv2(features)
        return features + x

    def _masked_forward(self, x, mask):
        features = self.bn1(x)
        features = torch.nn.functional.relu(features)
        features = self.conv1(features)
        features[mask.expand(-1, self.num_bottleneck_units, -1)] = 0
        features = self.bn2(features)
        features = torch.nn.functional.relu(features)
        features = self.conv2(features)
        features[mask.expand(-1, self.filters, -1)] = 0
        return features + x

    def forward(self, x, mask=None):
        if mask is None:
            return self._forward(x)
        else:
            return self._masked_forward(x, mask)


class ResNet1dSequencePool(pl.LightningModule):
    def __init__(self, learning_rate, log_interval, training=True, **kwargs):

        super(ResNet1dSequencePool, self).__init__()

        self.learning_rate = learning_rate
        self.log_interval = log_interval

        self.vocab_size = 20
        self.res_block_n_filters = 256
        self.feat_dim = 128
        self.res_block_kernel_size = 3
        self.n_res_blocks = 5
        self.res_bottleneck_factor = 1

        self.loss_func = SupConLoss()

        self._setup_layers()

        if training:
            self.save_hyperparameters()

    def _setup_layers(self):

        # encoder conv
        self.initial_conv = nn.Conv1d(
            in_channels=self.vocab_size,
            out_channels=self.res_block_n_filters,
            kernel_size=self.res_block_kernel_size,
            padding="same",
        )

        self.embedding_trunk = torch.nn.ModuleList()

        for layer_index in range(self.n_res_blocks):
            self.embedding_trunk.append(
                ResidualBlock(
                    self.res_block_n_filters,
                    self.res_bottleneck_factor,
                    self.res_block_kernel_size,
                    layer_index,
                    1,
                    dilation_rate=None,
                )
            )

    def _masked_forward(self, x, mask):
        """
        Before each convolution or batch normalization operation, we zero-out
        the features in any location is padded in the input
        sequence
        """
        x = self.initial_conv(x)

        for layer in self.embedding_trunk:
            x = layer(x, mask)
        # re-zero regions
        x[mask.expand(-1, self.res_block_n_filters, -1)] = 0
        # and do an aggregation operation
        # TODO: replace denominator of mean with the correct
        # sequence length. Also add two learnable params:
        # a power on the denominator and numerator
        return x.mean(axis=-1)

    def _forward(self, x):
        x = self.initial_conv(x)
        for layer in self.embedding_trunk:
            x = layer(x)
        return x.mean(axis=-1)

    def forward(self, x, mask=None):
        if mask is None:
            embeddings = self._forward(x)
        else:
            embeddings = self._masked_forward(x, mask)

        # always normalize for contrastive learning
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1, p=2)
        return embeddings

    def _shared_step(self, batch):
        features, masks, labels = batch

        embeddings = self.forward(features, masks)

        f1, f2 = torch.split(embeddings, embeddings.shape[0] // 2, dim=0)
        embeddings = torch.cat((f1.unsqueeze(1), f2.unsqueeze(1)), dim=1)

        loss = self.loss_func(embeddings, labels=labels)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                dots = torch.matmul(embeddings[:, 0], embeddings[:, 1].T)
                with torch.no_grad():
                    fig, ax = plt.subplots(ncols=1)
                    ax.imshow(
                        dots.to("cpu").numpy().astype(float), interpolation="nearest"
                    )
                    ax.set_title(
                        f"min: {torch.min(dots).item():.3f} max: {torch.max(dots).item()}"
                    )
                    self.logger.experiment.add_figure(
                        f"image", plt.gcf(), global_step=self.global_step
                    )
                    plt.close()

        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
