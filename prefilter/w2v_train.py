import os
import time
import pdb
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from glob import glob

import utils.utils as u
import models as m
import losses as l

try:
    from sklearn.metrics import confusion_matrix
except:
    print('cant import sklearn')

from pytorch_lightning.metrics import MetricCollection, Accuracy, Precision, Recall
from glob import glob
from argparse import ArgumentParser

if __name__ == '__main__': 

    root = '/home/tom/pfam-carbs/small-dataset/'
    train = glob(os.path.join(root, "*train.json"))
    test = glob(os.path.join(root, "*test-split.json"))

    loss_func = torch.nn.BCEWithLogitsLoss()
    args = m.PROT2VEC_CONFIG
    args['loss_func'] = loss_func
    args['test_files'] = test
    args['train_files'] = train
    args['valid_files'] = test
    args['normalize'] = True
    args['max_sequence_length'] = None
    args['lr'] = 1e-4
    args['batch_size'] = 8
    args['num_workers'] = 10
    args['gamma'] = 0.9
    args['n_negative_samples'] = 5

    model = m.Prot2Vec(args)
    num_epochs = 50

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(gpus=1, max_epochs=num_epochs,
            check_val_every_n_epoch=10,
            default_root_dir='/home/tom/Dropbox/',
            callbacks=[lr_monitor])

    trainer.fit(model)

    torch.save(model.state_dict(), 'with-normalization-small-dataset-50.pt')
