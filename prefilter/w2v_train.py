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
    ap = ArgumentParser()

    ap.add_argument('--log_dir', required=True)
    ap.add_argument('--gpus', required=False,
            type=int, default=1)
    ap.add_argument('--batch_size', required=False,
            type=int, default=8)
    ap.add_argument('--epochs', required=False,
            type=int, default=10)

    parser_args = ap.parse_args()
    log_dir = parser_args.log_dir

    root = '../data/small-dataset'
    train = glob(os.path.join(root, "*train.json"))
    test = glob(os.path.join(root, "*test-split.json"))

    loss_func = torch.nn.BCEWithLogitsLoss()

    args = m.PROT2VEC_CONFIG
    args['loss_func'] = loss_func
    args['test_files'] = test[:2]
    args['train_files'] = train
    args['valid_files'] = test[:2]

    args['normalize'] = True
    args['max_sequence_length'] = None
    args['lr'] = 1e-3
    args['batch_size'] = parser_args.batch_size
    args['num_workers'] = 10
    args['gamma'] = 0.9
    args['n_negative_samples'] = 5

    model = m.Prot2Vec(args)
    num_epochs = parser_args.epochs

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(gpus=parser_args.gpus,
                      max_epochs=num_epochs,
                      check_val_every_n_epoch=10,
                      default_root_dir=log_dir,
                      callbacks=[lr_monitor],
                      accelerator='ddp')

    trainer.fit(model)

    torch.save(model.state_dict(), 'with-normalization-small-dataset-all-1000-ep.pt')
