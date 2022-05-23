import pdb
import os
import esm
from abc import ABC

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl

import prefilter.models as model_utils
import prefilter.utils as utils
from pathlib import Path

__all__ = ["ResNet1d"]


class ResNet1d(pl.LightningModule, ABC):
    def __init__(self, learning_rate, embed_msas, training=True):

        super(ResNet1d, self).__init__()

        self.learning_rate = learning_rate
        self.training = training
        self.embed_msas = embed_msas

        if self.embed_msas:
            self.msa_transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
            self.msa_transformer.eval()
            self.msa_transformer.requires_grad_ = False

        if self.msa_transformer:
            self.res_block_n_filters = 768
            self.msa_mlp = torch.nn.Sequential(
                *[
                    torch.nn.Conv2d(
                        self.res_block_n_filters, self.res_block_n_filters, 1
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        self.res_block_n_filters, self.res_block_n_filters, 1
                    ),
                ]
            )
        else:
            self.res_block_n_filters = 256

        self.res_block_kernel_size = 3
        self.n_res_blocks = 12
        self.res_bottleneck_factor = 1
        self.padding = "same"
        self.padding_mode = "circular"

        self.log_interval = 100

        self.loss_func = model_utils.SupConLoss()

        self._setup_layers()

        self.save_hyperparameters()

    def _setup_layers(self):

        self.embed = nn.Embedding(27, self.res_block_n_filters)
        print(f"Using padding {self.padding_mode}, with {self.padding} padding.")

        _list = []
        for _ in range(self.n_res_blocks):
            _list.append(
                model_utils.ResConv(
                    self.res_block_n_filters,
                    kernel_size=self.res_block_kernel_size,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                )
            )
        self.embedding_trunk = torch.nn.Sequential(*_list)

        mlp_list = [
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.res_block_n_filters, self.res_block_n_filters, 1),
        ]
        self.mlp = torch.nn.Sequential(*mlp_list)

    def _forward(self, x):
        x = self.embed(x)
        x = self.embedding_trunk(x.transpose(-1, -2))
        x = self.mlp(x)
        return x

    def _masked_forward(self, x, mask):
        x = self.embed(x).transpose(-1, -2)
        x = ~mask[:, None, :] * x
        for layer in self.embedding_trunk:
            x, mask = layer.masked_forward(x, mask)
        # mask here
        # removed a transpose for the LSTM.
        # point-wise mlp; no downsampling.
        # no need to mask the mask (kernel size 1).
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
        if self.msa_transformer:
            if len(batch) == 3:
                msas, seqs, labels = batch
            else:
                msas, msa_labels, seqs, seq_labels, labels = batch
        elif len(batch) == 4:
            features, masks, labelvecs, labels = batch
        else:
            features, masks, labelvecs = batch

        if self.msa_transformer:
            # each msa gets an _entire_ embedding
            msa_embeddings = self.msa_transformer(
                msas, repr_layers=[12], return_contacts=False
            )["representations"][12].detach()
            msa_embeddings = msa_embeddings[:, :, 1:, :]
            # use a cnn to embed the MSAs
            # it's an image though... and we want a single final
            # embedding to use, so make the embedding dimension
            # the channel dimension
            # apply a CNN with embed dim. as channel dimension
            msa_embeddings = self.msa_mlp(msa_embeddings.transpose(-1, 1))
            # mean pool sequence embeddings across
            # msa dimension
            msa_embeddings = msa_embeddings.transpose(-1, 1).mean(dim=1)
            sequence_embeddings = self.forward(seqs).transpose(-1, -2)
            if self.global_step % self.log_interval == 0:
                with torch.no_grad():
                    _msa = torch.cat(torch.unbind(msa_embeddings, dim=0))
                    _seq = torch.cat(torch.unbind(sequence_embeddings, dim=0))
                    plt.imshow(torch.matmul(_msa, _seq.T).to("cpu").detach().numpy())
                    plt.colorbar()
                    fpath = (
                        f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",
                    )
                    if os.path.isdir(self.trainer.logger.log_dir):
                        print(f"saving to {fpath}")
                        plt.savefig(
                            f"{self.trainer.logger.log_dir}/image_{self.global_step}.png",
                            bbox_inches="tight",
                        )
                    plt.close()
        else:
            if masks is not None:
                embeddings, masks = self.forward(features, masks)
            else:
                embeddings = self.forward(features)

        if self.msa_transformer:
            msa_characters = []
            seq_characters = []

            # grab the aligned characters between the query sequence and the msa embedding and push them together.
            for msa_label, query_label, msa_embed, seq_embed in zip(
                msa_labels, seq_labels, msa_embeddings, sequence_embeddings
            ):
                # only grab 1 msa label
                msa_label = msa_label[0]
                # now i need to knock out the -1s... how do I do this?
                end = torch.sum(msa_label == -1)
                # put negative unique labels in masked positions
                msa_label[msa_label == -1] = torch.arange(
                    -1, -end - 1, step=-1, dtype=msa_label.dtype
                ).to(msa_label.device)
                query_label[query_label == -1] = torch.arange(
                    -end - 2,
                    -end - 2 - torch.sum(query_label == -1),
                    step=-1,
                    dtype=query_label.dtype,
                ).to(query_label.device)
                aligned = torch.where(msa_label == query_label.T)[0]
                msa_characters.extend(msa_embed[aligned])
                seq_characters.extend(seq_embed[aligned])

            msa_embeddings = torch.stack(msa_characters)
            sequence_embeddings = torch.stack(seq_characters)
            msa_embeddings = torch.nn.functional.normalize(msa_embeddings, dim=-1)
            sequence_embeddings = torch.nn.functional.normalize(
                sequence_embeddings, dim=-1
            )

            # have to normalize for the supcon loss
            loss = self.loss_func(
                torch.cat(
                    (msa_embeddings.unsqueeze(1), sequence_embeddings.unsqueeze(1)),
                    dim=1,
                )
            )
        else:
            embeddings, _ = torch.max(embeddings, dim=-1)
            e1, e2 = torch.split(embeddings, embeddings.shape[0] // 2, dim=0)
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
