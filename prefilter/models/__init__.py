import pytorch_lightning as pl
import torch.nn as nn

from .deepfam import *
from .deepnog import *
from .attn import *

class ClassificationTask(pl.LightningModule):

    def __init__(self, model, model_dict):
        super().__init__()
        
        self.train_metrics = model_dict['metrics'].clone()
        self.valid_metrics = model_dict['metrics'].clone()

        for key, val in model_dict.items():
            setattr(self, key, val) # lazy lazy

        self.model = model

        if self.multilabel_classification:
            self.class_act = nn.Sigmoid()
        else:
            self.class_act = nn.Softmax(dim=1)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.train_metrics(self.class_act(y_hat).ravel(), y.long().ravel())
        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.valid_metrics(self.class_act(y_hat).ravel(), y.long().ravel())
        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.valid_metrics(self.class_act(y_hat).ravel(), y.long().ravel())
        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):

        return self.optim(self.parameters(), lr=self.lr)


def configure_metrics():

    metric_collection = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        pl.metrics.Recall(),
        pl.metrics.Precision()
        ])

    return metric_collection
