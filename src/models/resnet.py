import pytorch_lightning as pl

from src.models.mean_pool import ResidualBlock


class ResNetParamFactory(pl.LightningModule):
    def __init__(
        self,
        n_res_blocks,
        res_block_n_filters,
    ):

        super(ResNetParamFactory, self).__init__()

        self.initial_conv = nn.Conv1d(
            in_channels=20,
            out_channels=res_block_n_filters,
            kernel_size=3,
            padding="same",
        )

        self.embedding_trunk = torch.nn.ModuleList()

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

    def forward(self, x):
        return self.embedding_trunk(self.initial_conv(x))

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
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )
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
