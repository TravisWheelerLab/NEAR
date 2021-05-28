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

    root = '../data/pmark-outputs/profmark0.7/json/'
    max_sequence_length = 256
    name_to_label_mapping = 
            '../../data/pmark-outputs/profmark0.6/json/name-to-label.json')
    train = u.Word2VecStyleDataset(root + 'train-sequences-and-labels.json')
    test = u.Word2VecStyleDataset(root + 'test-sequences-and-labels.json')

    model = m.Prot2Vec(m.PROTCNN_CONFIG, {})
    args = {}
    print(model)







