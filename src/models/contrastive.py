import pdb

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.utils.layers import ResConv
from src.losses import NpairLoss, SupConLoss


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
        indels=False,
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

        self.loss_func = NpairLoss()

        self._setup_layers()

        self.save_hyperparameters()
<<<<<<< HEAD
        self.training_step_outputs = []
        self.validation_step_outputs = []
=======
>>>>>>> main

    def _setup_layers(self):

        self.embed = nn.Conv1d(
<<<<<<< HEAD
            in_channels=self.in_channels,
            out_channels=self.res_block_n_filters,
            kernel_size=1,
=======
            in_channels=self.in_channels, out_channels=self.res_block_n_filters, kernel_size=1,
>>>>>>> main
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
        l1 = torch.cat(torch.unbind(feature1_indices, dim=0))
        l2 = torch.cat(torch.unbind(feature2_indices, dim=0))
        labelmat = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()

        return labelmat

    def _shared_step(self, batch):

        if self.indels:
<<<<<<< HEAD
            (
                seq1,
                labels1,
                seq2,
                labels2,
            ) = batch  # 32 pairs of sequences, each amino has a label

            features = torch.cat([seq1, seq2], dim=0)

=======
            (seq1, labels1, seq2, labels2,) = batch  # 32 pairs of sequences, each amino has a label

            features = torch.cat([seq1, seq2], dim=0)

            # mask = self.construct_mask(labels1, labels2)
>>>>>>> main
            seq_len = labels1.shape[1]

            # have to make the labels unique for every batchs
            for batch_idx in range(labels1.shape[0]):
                labels1[batch_idx] += batch_idx * seq_len
                labels2[batch_idx] += batch_idx * seq_len

            l1 = torch.cat(torch.unbind(labels1, dim=0))
            l2 = torch.cat(torch.unbind(labels2, dim=0))
            mask = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()
            e1_indices = torch.where(~l1.isnan())[0]
            e2_indices = torch.where(~l2.isnan())[0]
        else:
            features = batch
            mask = e1_indices = e2_indices = None

        embeddings = self.forward(features)
        # batch_size x sequence_length x embedding_dimension

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

        if self.global_step % self.log_interval == 0:
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 10))
                arr = torch.matmul(e1, e2.T).to("cpu").detach().numpy()
                arr = arr.astype(float)
                plt.imshow(arr)
                plt.colorbar()
                self.logger.experiment.add_figure(f"image", plt.gcf(), global_step=self.global_step)
<<<<<<< HEAD

=======
        # loss = self.loss_func(
        #     torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1), mask = mask
        # )
>>>>>>> main
        # input is ((batch_size/2) x 2 x embedding_dimension)

        loss = self.loss_func(
            e1, e2, mask, e1_indices, e2_indices
        )  # input is ((batch_size/2) x 2 x embedding_dimension)
        return loss

    def training_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
<<<<<<< HEAD
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self._shared_step(batch)
        self.validation_step_outputs.append(loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()  # free memory
=======
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
>>>>>>> main

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

<<<<<<< HEAD
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()  # free memory
=======
    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        self.log("val_loss", val_loss)
>>>>>>> main


class ResNet1dMultiPos(ResNet1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = SupConLoss()

    def _shared_step(self, batch):

        (features, labels) = batch  # 32 pairs of sequences, each amino has a label
        # these are now not all the same size so we need to first relabel and then flatten
<<<<<<< HEAD
=======

        # #now flatten along the label idx dimension
        # loss = self.loss_func(features, labels)

        # l1 = torch.cat(torch.unbind(labels1, dim=0))
        # l2 = torch.cat(torch.unbind(labels2, dim=0))
        # mask = torch.eq(l1.unsqueeze(1), l2.unsqueeze(0)).float()
        # e1_indices = torch.where(~l1.isnan())[0]
        # e2_indices = torch.where(~l2.isnan())[0]

>>>>>>> main
        embeddings = self.forward(features)
        # batch_size x sequence_length x embedding_dimension

        embeddings_transposed = embeddings.transpose(
            -1, -2
        )  # batch_size x sequence_length x embedding_dimension
<<<<<<< HEAD

        e = torch.cat(torch.unbind(embeddings_transposed, dim=0))

        en = torch.nn.functional.normalize(e, dim=-1)
        l = torch.stack(labels).flatten()
        loss = self.loss_func(en.unsqueeze(1), l)

=======
        e = torch.cat(torch.unbind(embeddings_transposed, dim=0)).unsqueeze(1)
        l = torch.stack(labels).flatten()
        loss = self.loss_func(e, l)

        # e1, e2 = torch.split(
        #     embeddings_transposed,
        #     embeddings.shape[0] // 2,
        #     dim=0,  # both are (batch_size /2 , sequence_length, embedding_dimension)
        # )  # -- see datasets collate_fn
        # e1 = torch.cat(torch.unbind(e1, dim=0))  # original seq embeddings
        # e2 = torch.cat(torch.unbind(e2, dim=0))  # mutated seq embeddings
        # # ((batch_size/2) * sequence_length) x embedding_dimension

        # if self.global_step % self.log_interval == 0:
        #     with torch.no_grad():
        #         fig = plt.figure(figsize=(10, 10))
        #         arr = torch.matmul(e1, e2.T).to("cpu").detach().numpy()
        #         arr = arr.astype(float)
        #         plt.imshow(arr)
        #         plt.colorbar()
        #         self.logger.experiment.add_figure(
        #             f"image", plt.gcf(), global_step=self.global_step
        #         )
        # # loss = self.loss_func(
        # #     torch.cat((e1.unsqueeze(1), e2.unsqueeze(1)), dim=1), mask = mask
        # # )
        # # input is ((batch_size/2) x 2 x embedding_dimension)

        # loss = self.loss_func(
        #     e1, e2, mask, e1_indices, e2_indices
        # )  # input is ((batch_size/2) x 2 x embedding_dimension)
>>>>>>> main
        return loss
