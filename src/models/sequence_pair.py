import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.layers import ResConv


class SequencePairClassifier(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        log_interval,
        res_block_n_filters=256,
        res_block_kernel_size=3,
        n_res_blocks=8,
        training=True,
    ):

        super(SequencePairClassifier, self).__init__()

        self.learning_rate = learning_rate
        self.training = training
        self.res_block_n_filters = res_block_n_filters
        self.res_block_kernel_size = res_block_kernel_size
        self.n_res_blocks = n_res_blocks
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "circular"

        self.log_interval = log_interval

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.embed = nn.Conv1d(
            in_channels=40, out_channels=self.res_block_n_filters, kernel_size=1,
        )

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
        self.final = nn.Conv1d(
            # what k-mer information is the most informative?
            in_channels=self.res_block_n_filters * self.n_res_blocks,
            out_channels=self.res_block_n_filters,
            kernel_size=1,
        )
        self.linear = nn.Linear(self.res_block_n_filters, 1)

        self.embedding_trunk = torch.nn.Sequential(*_list)

    def _forward(self, x):
        x = self.embed(x)
        activations = []
        for layer in self.embedding_trunk:
            x = layer(x)
            activations.append(x)
        # concatenate along embedding dimension:
        # cat/sum? x
        x = self.final(torch.nn.ReLU()(torch.cat(activations, dim=1)))
        return x

    def _masked_forward(self, x, mask):
        x = self.embed(x).transpose(-1, -2)
        x = ~mask[:, None, :] * x
        for layer in self.embedding_trunk:
            x, mask = layer.masked_forward(x, mask)

        x = self.mlp(x)
        x = ~mask[:, None, :] * x
        return x, mask

    def forward(self, x, masks=None):
        if masks is not None:
            embeddings, masks = self._masked_forward(x, masks)
            return embeddings, masks
        else:
            embeddings = self._forward(x)
            return embeddings

    def _shared_step(self, batch):
        f1, f2, labels = batch
        # concatenate pairs along embedding dimension
        true_embeddings = self.forward(torch.cat((f1, f2), dim=1)).mean(dim=-1)
        # reverse along the batch dimension so we have a bunch of negative pairs
        false_embeddings = self.forward(torch.cat((f1, torch.flip(f2, dims=(0,))), dim=1)).mean(
            dim=-1
        )

        classified = self.linear(torch.cat((true_embeddings, false_embeddings), dim=0))
        labels = torch.ones_like(classified)
        labels[true_embeddings.shape[0] :] = 0
        loss = self.loss_func(classified, labels)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                acc = (
                    torch.sum(labels == torch.round(torch.sigmoid(classified)))
                    / classified.shape[0]
                )
                self.log("acc", acc)

        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
