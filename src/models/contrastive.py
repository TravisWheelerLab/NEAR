import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.layers import ResConv
from src.utils.losses import SupConLoss
import pdb

class ResNet1d(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        log_interval,
        res_block_n_filters: int = 256,
        res_block_kernel_size: int = 3,
        in_channels: int = 128,
        n_res_blocks: int = 8,
        training: bool = True,
        indels = False
    ):

        super(ResNet1d, self).__init__()

        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.training = training
        self.res_block_n_filters = res_block_n_filters
        self.res_block_kernel_size = res_block_kernel_size
        self.n_res_blocks = n_res_blocks
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "circular"
        self.indels = indels

        self.log_interval = log_interval

        self.loss_func = SupConLoss()

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.embed = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.res_block_n_filters,
            kernel_size=1,
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
            in_channels=self.res_block_n_filters,
            out_channels=self.res_block_n_filters,
            kernel_size=1,
        )

        self.embedding_trunk = torch.nn.Sequential(*_list)

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x)
        x = self.final(x)
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

    def construct_mask(self, feature1_indices, feature2_indices):
        seq_len = feature1_indices.shape[1]

        for batch_idx in range(feature1_indices.shape[0]):
            feature1_indices[batch_idx] += batch_idx * seq_len
            feature2_indices[batch_idx] += batch_idx * seq_len
        #feature1_indices = feature1_indices.contiguous().view(-1,1)
        #feature2_indices = feature2_indices.contiguous().view(-1,1)
        #mask = torch.eq(feature1_indices, feature2_indices.T)
        l1 = torch.cat(torch.unbind(feature1_indices, dim=0))
        l2 = torch.cat(torch.unbind(feature2_indices, dim=0))
        labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()

        return labelmat

    def _shared_step(self, batch):

        if self.indels:
            (
                seq1,
                labels1,
                seq2,
                labels2,
            ) = batch  # 32 pairs of sequences, each amino has a label

            features = torch.cat([seq1, seq2], dim = 0)
        
            mask = self.construct_mask(labels1, labels2)
        else:
            features = batch
            mask = None

        embeddings = self.forward(features)
        # batch_size x sequence_length x embedding_dimension
        # 32x768x200

        embeddings_transposed = embeddings.transpose(
            -1, -2
        )  # batch_size x sequence_length x embedding_dimension

        e1, e2 = torch.split(
            embeddings_transposed,
            embeddings.shape[0] // 2,
            dim=0,  # both are (batch_size /2 , sequence_length, embedding_dimension)
        )  # -- see datasets collate_fn
        e1 = torch.cat(torch.unbind(e1, dim=0))  # original seq embeddings
        e2 = torch.cat(torch.unbind(e2, dim=0))  # mutated seq embeddings
        # ((batch_size/2) * sequence_length) x embedding_dimension

        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                arr = torch.matmul(e1, e2.T).to("cpu").detach().numpy()
                arr = arr.astype(float)
                plt.imshow(arr)
                plt.colorbar()
                self.logger.experiment.add_figure(f"image", plt.gcf(), global_step=self.global_step)
        loss = self.loss_func(
            torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1), mask = mask
        )  # input is ((batch_size/2) x 2 x embedding_dimension)

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


class ResNet1dKmerSampler(ResNet1d):
    def __init__(self, *args, **kwargs):
        super(ResNet1dKmerSampler, self).__init__(*args, **kwargs)

    def _shared_step(self, batch):
        features, labels = batch

        embeddings = self.forward(features)

        e1, e2 = torch.split(embeddings.mean(dim=-1), embeddings.shape[0] // 2, dim=0)
        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                arr = torch.matmul(e1, e2.T).to("cpu").detach().numpy()
                arr = arr.astype(float)
                plt.imshow(arr)
                plt.colorbar()
                self.logger.experiment.add_figure(f"image", plt.gcf(), global_step=self.global_step)

        loss = self.loss_func(torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1))

        return loss


class ResNet1dKmerSamplerWithLabelVectors(ResNet1d):
    def __init__(self, *args, **kwargs):
        super(ResNet1dKmerSamplerWithLabelVectors, self).__init__(*args, **kwargs)

    def _shared_step(self, batch):
        features, labels = batch

        embeddings = self.forward(features)

        e1, e2 = torch.split(embeddings.mean(dim=-1), embeddings.shape[0] // 2, dim=0)
        e1 = torch.nn.functional.normalize(e1, dim=-1)
        e2 = torch.nn.functional.normalize(e2, dim=-1)

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                arr = torch.matmul(e1, e2.T).to("cpu").detach().numpy()
                arr = arr.astype(float)
                plt.imshow(arr)
                plt.colorbar()
                self.logger.experiment.add_figure(f"image", plt.gcf(), global_step=self.global_step)

        loss = self.loss_func(torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1))

        return loss
