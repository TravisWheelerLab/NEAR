# pylint: disable=no-member
import pdb

import pytorch_lightning as pl
import io
import pandas as pd
import torchmetrics
import torch
import numpy as np
import torchvision
import prefilter.utils as utils
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from wandb import Table
from collections import defaultdict

__all__ = ["BaseModel"]


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):

        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.sigmoid_threshold_to_tps_passed = defaultdict(int)
        self.sigmoid_threshold_to_num_passed = defaultdict(int)
        self.sigmoid_threshold_to_decoys_passed = defaultdict(int)
        self.sigmoid_threshold_to_tps_missed = defaultdict(int)
        self.sigmoid_threshold_to_fps_passed = defaultdict(int)
        self.thresholds = range(10, 101, 5)[::-1]
        self.thresholds = [t / 100 for t in self.thresholds]
        self.total_sequences = 0
        self.total_true_labels = 0
        self.val_confusion = None
        self.train_confusion = None

    def _create_datasets(self):
        # This will be shared between every model that I train.
        self.train_dataset = utils.ProteinSequenceDataset(
            self.train_files,
            single_label=self.single_label,
            sample_sequences_based_on_family_membership=self.resample_families,
            sample_sequences_based_on_num_labels=self.resample_based_on_num_labels,
            resample_based_on_uniform_dist=self.resample_based_on_uniform_dist,
            use_pretrained_model_embeddings=not self.train_from_scratch,
        )

        self.val_dataset = utils.ProteinSequenceDataset(
            self.val_files,
            single_label=self.single_label,
            existing_name_to_label_mapping=self.train_dataset.name_to_class_code,
            use_pretrained_model_embeddings=not self.train_from_scratch,
        )

        self.val_and_decoy_dataset = utils.ProteinSequenceDataset(
            self.val_files,
            single_label=self.single_label,
            existing_name_to_label_mapping=self.val_dataset.name_to_class_code,
            use_pretrained_model_embeddings=not self.train_from_scratch,
        )

        self.class_code_mapping = self.val_dataset.name_to_class_code
        self.n_classes = self.val_dataset.n_classes
        self.val_confusion = torchmetrics.ConfusionMatrix(num_classes=self.n_classes)
        self.train_confusion = torchmetrics.ConfusionMatrix(num_classes=self.n_classes)

    def _setup_layers(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel"
            "must implement the _setup_layers()"
            "method"
        )

    def _forward(self):
        raise NotImplementedError()

    def _masked_forward(self):
        raise NotImplementedError()

    def forward(self, x, mask=None):
        raise NotImplementedError()

    def _shared_step(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_nb):
        loss, acc, logits, labels = self._shared_step(batch)
        self.train_confusion.update(logits, labels)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, acc, logits, labels = self._shared_step(batch)
        self.val_confusion.update(logits, labels)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        if self.schedule_lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.step_lr_step_size,
                    gamma=self.step_lr_decay_factor,
                ),
            }
        else:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_epoch_end(self, outputs):
        train_loss = self.all_gather([x["loss"] for x in outputs])
        train_acc = self.all_gather([x["train_acc"] for x in outputs])
        loss = torch.mean(torch.stack(train_loss))
        acc = torch.mean(torch.stack(train_acc))
        self.log("train/loss", loss)
        self.log("train/acc", acc)
        self.log("learning_rate", self.learning_rate)
        self.log_cmat(self.train_confusion, "train/cmat")

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

    def log_cmat(self, cmat, tag):
        conf_mat = cmat.compute().detach().cpu().numpy().astype(np.int)
        df_cm = pd.DataFrame(
            conf_mat, index=np.arange(self.n_classes), columns=np.arange(self.n_classes)
        )

        fig = plt.figure(figsize=(12, 10), dpi=300)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 1}, fmt="d")
        self.logger.experiment.add_figure(tag, fig, global_step=self.global_step)
        plt.close(fig)
        cmat.reset()

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_acc = self.all_gather([x["val_acc"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        val_acc = torch.mean(torch.stack(val_acc))
        self.log("val/loss", val_loss)
        self.log("val/acc", val_acc)
        self.log_cmat(self.val_confusion, "val/cmat")

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size == 1 else utils.pad_batch,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size == 1 else utils.pad_batch,
        )
        return val_loader

    def test_dataloader(self):
        val_and_decoy_loader = torch.utils.data.DataLoader(
            self.val_and_decoy_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=None if self.batch_size == 1 else utils.pad_batch,
        )
        return val_and_decoy_loader

    def on_test_epoch_end(self):
        percent_tps_recovered = [
            [x, y / self.total_true_labels]
            for x, y in zip(
                self.thresholds, self.sigmoid_threshold_to_tps_passed.values()
            )
        ]
        table = Table(
            data=percent_tps_recovered, columns=["threshold", "tps_recovered"]
        )
        self.logger.experiment.log(
            {
                "percent_tps_recovered": wandb.plot.line(
                    table,
                    "threshold",
                    "tps_recovered",
                    stroke=None,
                    title="tps recovered",
                )
            }
        )
        mean_fps_per_sequence = [
            [x, y / self.total_sequences]
            for x, y in zip(
                self.thresholds, self.sigmoid_threshold_to_fps_passed.values()
            )
        ]
        table = Table(
            data=mean_fps_per_sequence, columns=["threshold", "fps_per_sequence"]
        )
        self.logger.experiment.log(
            {
                "fps_per_sequence": wandb.plot.line(
                    table,
                    "threshold",
                    "fps_per_sequence",
                    stroke=None,
                    title="fps per sequence",
                )
            }
        )

    def test_step(self, batch, batch_nb):
        features, masks, labels = batch
        self.total_sequences += features.shape[0]
        self.total_true_labels += torch.count_nonzero(labels)
        logits = self.forward(features, masks)
        scores = self.class_act(logits)
        for threshold in self.thresholds:
            scores[scores >= threshold] = 1
            # TODO: do I need to include decoys?
            true_positives = torch.sum(
                torch.count_nonzero((scores == 1).bool() & (labels == 1).bool())
            )
            false_positives = torch.sum(
                torch.count_nonzero((scores == 1).bool() & (labels == 0).bool())
            )
            misses = torch.sum(
                torch.count_nonzero((scores != 1).bool() & (labels == 1).bool())
            )
            num_passed = torch.sum(torch.count_nonzero(scores == 1))
            self.sigmoid_threshold_to_num_passed[threshold] += num_passed
            self.sigmoid_threshold_to_tps_missed[threshold] += misses
            self.sigmoid_threshold_to_tps_passed[threshold] += true_positives
            self.sigmoid_threshold_to_fps_passed[threshold] += false_positives