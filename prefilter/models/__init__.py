import pytorch_lightning as pl
import torch.nn as nn
import torch
import pandas as pd
import numpy as np

from .deepfam import *
from .deepnog import *
from .attn import *
from .protcnn import *

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    good_import_matplotlib = True
except:
    good_import_matplotlib = False
    print('couldn\'t import matplotlib')


class ClassificationTask(pl.LightningModule):

    def __init__(self, model, args):
        super().__init__()
        
        self.train_metrics = args['metrics'].clone()
        self.valid_metrics = args['metrics'].clone()
        self.test_metrics = args['metrics'].clone()
        
        for key, val in args.items():
            setattr(self, key, val)

        self.train_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.valid_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.test_confmat = pl.metrics.ConfusionMatrix(num_classes=2)

        if self.threshold_curve:
            self.test_prcurve =  PRCurve(self.device)
            self.train_prcurve =  PRCurve(self.device)
            self.valid_prcurve =  PRCurve(self.device)

        self.model = model
        self.save_hyperparameters()

        if self.multilabel:
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

        self.log_dict(self.train_metrics)
        self.log('train loss', loss)

        if self.log_freq == self.current_epoch and self.threshold_curve:
            self.train_prcurve.update(preds, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        preds = self.class_act(y_hat).ravel() 
        y = y.long().ravel()

        self.valid_metrics(preds, y)
        self.valid_confmat.update(preds, y)

        self.log_dict(self.valid_metrics)
        self.log('valid loss', loss)

        if self.log_freq == self.current_epoch and self.threshold_curve:
            self.valid_prcurve.update(preds, y)

        return loss

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        preds = self.class_act(y_hat).ravel() 
        y = y.long().ravel()

        self.test_metrics(preds, y)
        self.test_confmat.update(preds, y)

        if self.threshold_curve:
            self.test_prcurve.update(preds, y)

        self.log_dict(self.test_metrics)
        self.log('test loss', loss)

        return loss

    # values returned by this func are stored over an epoch

    def train_epoch_end(self, outs):

        cmat = self.train_confmat.compute().detach().cpu().numpy().astype(np.int)
        if good_import_matplotlib:
            self._log_cmat(cmat, 'train_cmat')
            if self.threshold_curve and self.log_freq == self.current_epoch:
                th, ps, rs = self.train_prcurve.compute()
                self._log_threshold_curve(th, ps, rs, 'train')

    def validation_epoch_end(self, outs):

        cmat = self.valid_confmat.compute().detach().cpu().numpy().astype(np.int)
        if good_import_matplotlib:
            self._log_cmat(cmat, 'val_cmat')
            if self.threshold_curve and self.log_freq == self.current_epoch:
                th, ps, rs = self.valid_prcurve.compute()
                self._log_threshold_curve(th, ps, rs, 'valid')

    def test_epoch_end(self, outs):
        cmat = self.test_confmat.compute().detach().cpu().numpy().astype(np.int)
        if good_import_matplotlib:
            self._log_cmat(cmat, 'test_cmat')
            if self.threshold_curve:
                th, ps, rs = self.test_prcurve.compute()
                self._log_threshold_curve(th, ps, rs, 'test')

    def _log_cmat(self, confmat, name):

        df_cm = pd.DataFrame(confmat, index = range(2), columns=range(2))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure(name, fig_, self.current_epoch)

    def _log_threshold_curve(self, th, ps, rs, name):
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(th, ps, label='precision')
        plt.scatter(th, rs, label='recall')
        plt.ylim([0, 1])
        plt.legend()
        self.logger.experiment.add_figure('prec/rec' + name, fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):

        optim = self.optim(self.parameters(), lr=self.lr)
        schedule = torch.optim.lr_scheduler.StepLR(optim, step_size=self.step_size, gamma=self.gamma)
        return [optim], [schedule]



def configure_metrics():

    metric_collection = pl.metrics.MetricCollection([
        pl.metrics.Accuracy(),
        pl.metrics.Recall(),
        pl.metrics.Precision(),
        pl.metrics.classification.HammingDistance(),
        ])

    return metric_collection


class PRCurve(pl.metrics.Metric):
    ''' Use me on non-thresholded data (before rounding or argmax). '''

    def __init__(self, device, dist_sync_on_step=False):

        super().__init__(dist_sync_on_step=dist_sync_on_step)
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.precision_and_recall_accumulators = {}
        for th in thresholds:
            p = pl.metrics.Precision().to(device)
            r = pl.metrics.Recall().to(device)
            self.precision_and_recall_accumulators[th] = [p, r]

    def update(self, preds, target):

        for th, (precision, recall) in self.precision_and_recall_accumulators.items():
            p = preds.clone().cpu()
            t = target.cpu()
            p[p >= th] = 1
            precision.update(p, t)
            recall.update(p, t)

    def compute(self):

        th = list(self.precision_and_recall_accumulators.keys())
        ps = []
        rs = []
        for p, r in self.precision_and_recall_accumulators.values():
            ps.append(p.compute().cpu().detach().numpy())
            rs.append(r.compute().cpu().detach().numpy())

        return th, ps, rs
