import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np

from .deepfam import *
from .deepnog import *
from .attn import *

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    good_import_matplotlib = True
except:
    good_import_matplotlib = False


class ClassificationTask(pl.LightningModule):

    def __init__(self, model, model_dict):
        super().__init__()
        
        self.train_metrics = model_dict['metrics'].clone()
        self.valid_metrics = model_dict['metrics'].clone()
        self.test_metrics = model_dict['metrics'].clone()

        self.train_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.valid_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.test_confmat = pl.metrics.ConfusionMatrix(num_classes=2)

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
        preds = self.class_act(y_hat).ravel() 
        y = y.long().ravel()

        self.train_metrics(preds, y)
        self.train_confmat.update(preds, y)

        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
        self.log('train loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        preds = self.class_act(y_hat).ravel() 
        y = y.long().ravel()

        self.valid_metrics(preds, y)
        self.valid_confmat.update(preds, y)

        self.log_dict(self.valid_metrics, on_step=True, on_epoch=False)
        self.log('valid loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        preds = self.class_act(y_hat).ravel() 
        y = y.long().ravel()

        self.test_metrics(preds, y)
        self.test_confmat.update(preds, y)

        self.log_dict(self.test_metrics, on_step=True, on_epoch=False)
        self.log('test loss', loss)

        return loss # values returned by this func are stored over an epoch

    def train_epoch_end(self, outs):

        cmat = self.train_confmat.compute().detach().cpu().numpy().astype(np.int)
        if good_import_matplotlib:
            self._log_cmat(cmat, 'train_cmat')

    def validation_epoch_end(self, outs):

        cmat = self.valid_confmat.compute().detach().cpu().numpy().astype(np.int)
        if good_import_matplotlib:
            self._log_cmat(cmat, 'val_cmat')

    def test_epoch_end(self, outs):

        cmat = self.test_confmat.compute().detach().cpu().numpy().astype(np.int)
        if good_import_matplotlib:
            self._log_cmat(cmat, 'test_cmat')

    def _log_cmat(self, confmat, name):

        df_cm = pd.DataFrame(confmat, index = range(2), columns=range(2))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure(name, fig_, self.current_epoch)

    def configure_optimizers(self):

        return self.optim(self.parameters(), lr=self.lr)


def configure_metrics():

    metric_collection = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        pl.metrics.Recall(),
        pl.metrics.Precision()
        ])

    return metric_collection
