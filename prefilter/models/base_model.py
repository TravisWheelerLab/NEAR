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
from typing import NamedTuple, List, Optional, Dict
from collections import defaultdict

__all__ = ["BaseModel"]


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        train_files: List[str],
        val_files: List[str],
        emission_files: Optional[List[str]],
        schedule_lr: bool,
        step_lr_step_size: int,
        step_lr_decay_factor: float,
        batch_size: int,
        num_workers: int,
        name_to_class_code: Dict[str, int],
        n_emission_sequences: int,
        distill: bool,
        xent: bool,
        decoy_files: List[str],
    ):

        super(BaseModel, self).__init__()

        self.learning_rate = learning_rate
        self.train_files = train_files
        self.val_files = val_files
        self.emission_files = emission_files
        self.decoy_files = decoy_files
        self.schedule_lr = schedule_lr
        self.step_lr_step_size = step_lr_step_size
        self.step_lr_decay_factor = step_lr_decay_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name_to_class_code = name_to_class_code
        self.n_emission_sequences = n_emission_sequences
        self.distill = distill
        self.xent = xent

    def _init_metrics(self):
        self.train_f1 = torchmetrics.F1()
        self.val_f1 = torchmetrics.F1()
        self.train_recall = torchmetrics.Recall()
        self.val_recall = torchmetrics.Recall()

    def _create_datasets(self):
        # This will be shared between every model that I train.
        if self.emission_files is not None:
            self.train_files = self.emission_files + self.train_files

        if self.decoy_files is not None:
            self.train_files = self.decoy_files + self.train_files

        self.train_dataset = utils.SequenceIterator(
            self.train_files,
            self.name_to_class_code,
            distillation_labels=self.distill,
        )

        self.train_dataset.shuffle()

        self.val_dataset = utils.SequenceIterator(
            self.val_files,
            name_to_class_code=self.name_to_class_code,
            distillation_labels=self.distill,
            evalue_threshold=1e-5,
        )

        self.n_classes = len(self.name_to_class_code)

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
        loss, logits, labels = self._shared_step(batch)
        self.train_f1.update(logits, labels.int())
        self.train_recall.update(logits, labels.int())
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        loss, logits, labels = self._shared_step(batch)
        self.val_f1.update(logits, labels.int())
        self.val_recall.update(logits, labels.int())
        return {"val_loss": loss}

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
        loss = torch.mean(torch.stack(train_loss))
        self.log("train/loss", loss)
        self.log("learning_rate", self.learning_rate)
        self.log("train/f1", self.train_f1.compute())
        self.log("train/recall", self.train_recall.compute())
        self.train_dataset.shuffle()

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        self.log("val/loss", val_loss)
        self.log("val/f1", self.val_f1.compute())
        self.log("val/recall", self.val_recall.compute())

    def train_dataloader(self):
        if self.batch_size == 1:
            collate_fn = None
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
