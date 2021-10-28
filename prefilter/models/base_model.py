import pytorch_lightning as pl
import prefilter.utils as utils
import torch

__all__ = ['BaseModel']


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):

        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _create_datasets(self):
        # This will be shared between every model that I train.
        self.train_dataset = utils.ProteinSequenceDataset(self.train_files,
                                                          sample_sequences_based_on_family_membership=self.resample_families,
                                                          sample_sequences_based_on_num_labels=self.resample_based_on_num_labels,
                                                          use_pretrained_model_embeddings=not self.train_from_scratch)

        self.val_dataset = utils.ProteinSequenceDataset(self.val_files,
                                                        existing_name_to_label_mapping=self.train_dataset.name_to_class_code,
                                                        use_pretrained_model_embeddings=not self.train_from_scratch)

        self.class_code_mapping = self.val_dataset.name_to_class_code
        self.n_classes = self.val_dataset.n_classes

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
        loss, acc = self._shared_step(batch)
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_nb):
        loss, acc = self._shared_step(batch)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        return loss

    def configure_optimizers(self):
        if self.schedule_lr:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return {'optimizer': optimizer,
                    'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_lr_step_size,
                                                                    gamma=self.step_lr_decay_factor)}
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

    def on_train_start(self):
        self.log("hp_metric", self.learning_rate)

    def validation_epoch_end(self, outputs):
        val_loss = self.all_gather([x["val_loss"] for x in outputs])
        val_acc = self.all_gather([x["val_acc"] for x in outputs])
        val_loss = torch.mean(torch.stack(val_loss))
        val_acc = torch.mean(torch.stack(val_acc))
        self.log("val/loss", val_loss)
        self.log("val/acc", val_acc)

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
