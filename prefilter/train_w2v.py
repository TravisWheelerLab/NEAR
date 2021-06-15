import os
import time
import pdb
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

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

    root = '../data/subset-for-overfitting/'
    name_to_label_mapping = root + 'name-to-label.json'
    train = root + 'train-subset.json'
    test = root + 'test-subset.json'

    loss_func = torch.nn.BCEWithLogitsLoss()
    args = m.PROT2VEC_CONFIG

    args['loss_func'] = loss_func
    args['test_files'] = test
    args['train_files'] = train
    args['valid_files'] = test
    args['max_sequence_length'] = None
    args['lr'] = 1e-2
    args['batch_size'] = 2
    args['num_workers'] = 1
    args['gamma'] = 1
    args['n_negative_samples'] = 5

    model = m.Prot2Vec(args)
    num_epochs = 1000
    trainer = Trainer(gpus=1, max_epochs=num_epochs,
            check_val_every_n_epoch=1000)
    trainer.fit(model)
    torch.save(model.state_dict(), 'overfit-on-subset.pt')

