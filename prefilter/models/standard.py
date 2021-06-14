import pytorch_lightning as pl
import torch.nn as nn
import torch
import pandas as pd
import numpy as np

from utils import utils as utils
from utils import datasets as datasets

__all__ = ['ClassificationTask', 'configure_metrics']

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    good_import_matplotlib = True
except:
    good_import_matplotlib = False
    print('couldn\'t import matplotlib')

class Word2VecTask(pl.LightningModule):

    def __init__(self, evaluating, args):

        super(Word2VecTask, self).__init__()
        
        for key, val in args.items():
            setattr(self, key, val) # easier than typing everything out

        if not evaluating:
            self.train_metrics = args['metrics'].clone()
            self.valid_metrics = args['metrics'].clone()
            self.test_metrics = args['metrics'].clone()
            self.train_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
            self.valid_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
            self.test_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
            self.save_hyperparameters(args)
            self._create_datasets()

        self.class_act = nn.Sigmoid()


    def forward(self, x):
        raise NotImplementedError()

    def _get_dots(self, targets, targets_mask, 
            in_context, in_context_mask,
            out_of_context,
            out_of_context_mask):

        targets_embed = self.forward(targets, targets_mask)
        context_embed = self.forward(in_context, in_context_mask)
        negatives_embed = self.forward(out_of_context, out_of_context_mask) 
        negatives_embed = torch.reshape(negatives_embed, (self.batch_size,
            self.n_negative_samples, self.embedding_dim))


        pos_dots = (targets_embed*context_embed).sum(axis=1).squeeze() 
        neg_dots = torch.bmm(negatives_embed, targets_embed.unsqueeze(2)).squeeze()
        return pos_dots, neg_dots.ravel()

    def _compute_loss_and_preds(self, batch):

        targets, targets_mask, contexts, contexts_mask, negatives, negatives_mask, y = batch

        pos_dots, neg_dots = self._get_dots(targets, targets_mask,
                contexts, contexts_mask,
                negatives, negatives_mask)

        y_hat = torch.cat((pos_dots, neg_dots), axis=0)
        y = y.ravel()

        loss = self.loss_func(y_hat, y) # should be binary xent
        preds = self.class_act(y_hat).ravel()
        y = y.long().ravel()

        return loss, preds, y, y_hat, pos_dots, neg_dots

    def training_step(self, batch, batch_idx):

        loss, preds, y, y_hat, pos_dots, neg_dots\
                = self._compute_loss_and_preds(batch)

        y_hat = y_hat.float()
        y = y.float()
        loss = self.loss_func(y_hat, y) # should be binary xent
        preds = self.class_act(y_hat).ravel()
        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds, y, y_hat, pos_dots, neg_dots\
                = self._compute_loss_and_preds(batch)

        self.valid_metrics(preds, y)
        self.valid_confmat.update(preds, y)

        self.log_dict(self.valid_metrics)
        self.log('valid loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        loss, preds, y, y_hat, pos_dots, neg_dots\
                = self._compute_loss_and_preds(batch)

        self.test_metrics(preds, y)
        self.test_confmat.update(preds, y)

        self.log_dict(self.test_metrics)
        self.log('test loss', loss)
        return loss

    # values returned by this func are stored over an epoch

    def _create_datasets(self):


        self.test_psd = datasets.Word2VecStyleDataset(self.test_files,
                                  self.max_sequence_length,
                                  self.name_to_label_mapping,
                                  evaluating=False,
                                  n_negative_samples=self.n_negative_samples
                                  )

        self.train_psd = datasets.Word2VecStyleDataset(self.train_files,
                                  self.max_sequence_length,
                                  self.name_to_label_mapping,
                                  evaluating=False,
                                  n_negative_samples=self.n_negative_samples
                                  )

        self.valid_psd = datasets.Word2VecStyleDataset(self.valid_files,
                                  self.max_sequence_length,
                                  self.name_to_label_mapping,
                                  evaluating=False,
                                  n_negative_samples=self.n_negative_samples
                                  )


    def train_dataloader(self):

        return torch.utils.data.DataLoader(self.train_psd, 
                batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=True,
                collate_fn=utils.pad_word2vec_batch)

    def test_dataloader(self):

        return torch.utils.data.DataLoader(self.test_psd,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=utils.pad_word2vec_batch)

    def val_dataloader(self):

        return torch.utils.data.DataLoader(self.valid_psd,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                collate_fn=utils.pad_word2vec_batch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def configure_optimizers_with_warmup(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # this is tuned for Adam 
        _, beta2 = optim.param_groups[0]['betas']
        # from ``on the adequacy of untuned warmup for adaptive optimization'''
        n_warmup_steps = 100 #int(2/(1-beta2))
        batches_per_epoch = len(self.train_dataloader())
        decay_steps = 20000

        def lr_schedule(step):

            if step < n_warmup_steps:
                x = self.lr * (step+1)/n_warmup_steps
                return x
            else:
                x = self.lr * self.gamma ** int(step / decay_steps)
                return x

        mysched = torch.optim.lr_scheduler.LambdaLR(optim, lr_schedule)
        sched = {'scheduler':mysched, 'interval':'step'} 
        return [optim], [sched]


class ClassificationTask(pl.LightningModule):

    def __init__(self, args):

        super(ClassificationTask, self).__init__()
        
        self.train_metrics = args['metrics'].clone()
        self.valid_metrics = args['metrics'].clone()
        self.test_metrics = args['metrics'].clone()
        
        for key, val in args.items():
            setattr(self, key, val) # easier than typing everything out

        self.train_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.valid_confmat = pl.metrics.ConfusionMatrix(num_classes=2)
        self.test_confmat = pl.metrics.ConfusionMatrix(num_classes=2)

        if self.threshold_curve:
            self.test_prcurve =  PRCurve(self.device)
            self.train_prcurve =  PRCurve(self.device)
            self.valid_prcurve =  PRCurve(self.device)

        self.save_hyperparameters(args)

        if self.multilabel:
            self.class_act = nn.Sigmoid()
        else:
            self.class_act = nn.Softmax(dim=1)

        self._create_datasets()


    def forward(self, x):
        raise NotImplementedError()


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
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
        y_hat = self.forward(x)
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
        y_hat = self.forward(x)
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

    def _create_datasets(self):

        self.test_psd = u.ProteinSequenceDataset(self.test_files,
                                  self.max_sequence_length,
                                  self.encode_as_image,
                                  self.multilabel,
                                  self.name_to_label_mapping,
                                  self.n_classes)

        self.train_psd = u.ProteinSequenceDataset(self.train_files,
                                  self.max_sequence_length,
                                  self.encode_as_image,
                                  self.multilabel,
                                  self.name_to_label_mapping,
                                  self.n_classes)

        self.valid_psd = u.ProteinSequenceDataset(self.valid_files,
                                  self.max_sequence_length,
                                  self.encode_as_image,
                                  self.multilabel,
                                  self.name_to_label_mapping,
                                  self.n_classes)


    def train_dataloader(self):

        return torch.utils.data.DataLoader(self.train_psd, batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):

        return torch.utils.data.DataLoader(self.test_psd, batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):

        return torch.utils.data.DataLoader(self.valid_psd, batch_size=self.batch_size,
                num_workers=self.num_workers, drop_last=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        # this is tuned for Adam 
        _, beta2 = optim.param_groups[0]['betas']
        # from ``on the adequacy of untuned warmup for adaptive optimization'''
        n_warmup_steps = 100 #int(2/(1-beta2))
        batches_per_epoch = len(self.train_dataloader())
        decay_steps = 20000

        def lr_schedule(step):

            if step < n_warmup_steps:
                x = self.lr * (step+1)/n_warmup_steps
                return x
            else:
                x = self.lr * self.gamma ** int(step / decay_steps)
                return x

        mysched = torch.optim.lr_scheduler.LambdaLR(optim, lr_schedule)
        sched = {'scheduler':mysched, 'interval':'step'} 
        return [optim], [sched]


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
            p = preds.clone().cpu() # i don't know why it has to be on CPU 
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

