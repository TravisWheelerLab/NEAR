# pylint: disable=no-member
import pdb

import pytorch_lightning as pl
import io
import pandas as pd
import torchmetrics
import torch
import numpy as np
import prefilter.utils as utils
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
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
        # self.train_f1 = torchmetrics.F1(ignore_index=0)
        # self.val_f1 = torchmetrics.F1(ignore_index=0)
        # self.accuracy = torchmetrics.Accuracy(ignore_index=0)
        self.train_f1 = torchmetrics.F1()
        self.val_f1 = torchmetrics.F1()
        self.accuracy = torchmetrics.Accuracy()

    def _create_datasets(self):
        # This will be shared between every model that I train.
        self.train_dataset = utils.ProteinSequenceDataset(
            self.train_files,
            self.name_to_class_code,
            self.n_seq_per_fam,
            single_embedding=not self.fcnn,
        )

        self.val_dataset = utils.SimpleSequenceIterator(
            self.val_files,
            name_to_class_code=self.name_to_class_code,
            single_embedding=not self.fcnn,
        )

        self.val_and_decoy_dataset = utils.SimpleSequenceIterator(
            self.val_files,
            name_to_class_code=self.name_to_class_code,
            single_embedding=not self.fcnn,
        )

        self.name_to_class_code = self.val_dataset.name_to_class_code
        self.n_classes = len(self.name_to_class_code)

        if self.log_confusion_matrix:

            self.val_confusion = torchmetrics.ConfusionMatrix(
                num_classes=self.n_classes
            )
            self.train_confusion = torchmetrics.ConfusionMatrix(
                num_classes=self.n_classes
            )

    def _setup_layers(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel"
            "must implement the _setup_layers()"
            "method"
        )

    def _forward(self, x):
        raise NotImplementedError()

    def _masked_forward(self, x, mask):
        raise NotImplementedError()

    def forward(self, x, mask=None):
        raise NotImplementedError()

    def _shared_step(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_nb):
        loss, acc, logits, labels = self._shared_step(batch)
        if self.log_confusion_matrix:
            self.train_confusion.update(logits, labels)
        self.train_f1.update(logits, labels)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, acc, logits, labels = self._shared_step(batch)
        if self.log_confusion_matrix:
            self.val_confusion.update(logits, labels)
        self.val_f1.update(logits, labels)
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
        if self.log_confusion_matrix:
            self.log_cmat(self.train_confusion, "train/cmat")
        self.log("train/f1", self.train_f1.compute())
        self.train_dataset.label_to_sequence.shuffle()

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
        if self.log_confusion_matrix:
            self.log_cmat(self.val_confusion, "val/cmat")
        self.log("val/f1", self.val_f1.compute())

    def train_dataloader(self):
        if self.batch_size == 1:
            collate_fn = None
        elif self.fcnn:
            collate_fn = utils.pad_labels_and_features_in_batch
        else:
            collate_fn = utils.pad_features_in_batch

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        if self.batch_size == 1:
            collate_fn = None
        elif self.fcnn:
            collate_fn = utils.pad_labels_and_features_in_batch
        else:
            collate_fn = utils.pad_features_in_batch

        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return val_loader
