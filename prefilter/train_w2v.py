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

    pmark = 0.7
    root = '../data/subset-for-overfitting/json/'
    max_sequence_length = 256
    name_to_label_mapping = root + 'name-to-label.json'
    train = root + 'test-subset.json'
    test = root + 'test-subset.json'
    valid = root + 'test-subset.json'

    loss_func = torch.nn.BCEWithLogitsLoss()
    arg_dict = {}
    arg_dict['metrics'] = m.configure_metrics()
    arg_dict['loss_func'] = loss_func
    arg_dict['test_files'] = test
    arg_dict['train_files'] = train
    arg_dict['valid_files'] = valid
    arg_dict['max_sequence_length'] = max_sequence_length
    arg_dict['name_to_label_mapping'] = name_to_label_mapping
    arg_dict['lr'] = 1e-1
    arg_dict['batch_size'] = 32
    arg_dict['num_workers'] = 1
    arg_dict['gamma'] = 1

    model = m.Prot2Vec(m.PROT2VEC_CONFIG, arg_dict)
    num_epochs = 100
    trainer = Trainer(gpus=1, max_epochs=num_epochs)
    trainer.fit(model)
    torch.save(model.state_dict(), 'overfit-on-subset.pt')

