import os

from pytorch_lightning.metrics import functional as FM

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
import time
import pdb
import sys
import pickle
import random

random.seed(1)

import torch
import pytorch_lightning as pl

import utils as utils
import models as m
import losses as l

import matplotlib.pyplot as plt
import numpy as np


from random import shuffle
from sklearn.metrics import confusion_matrix
from glob import glob
from functools import partial, update_wrapper
from argparse import ArgumentParser


n_classes = 17646 


if __name__ == '__main__':

    ap = ArgumentParser()

    model_group = ap.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--deepfam', action='store_true')
    model_group.add_argument('--deepnog', action='store_true')
    model_group.add_argument('--attn', action='store_true')

    loss_group = ap.add_mutually_exclusive_group(required=True)
    loss_group.add_argument('--binary-multilabel', action='store_true',
            help='sigmoid activation with N_CLASSES nodes on the last layer,\
            useful for doing multi-label classification')

    loss_group.add_argument('--multiclass',
            type=int, help='multiclass classification. Each sequence classified\
            (w/ softmax activation) into one of N_CLASSES')

    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', required=False, default=16, type=int)
    ap.add_argument('--max-sequence-length', required=False, default=256,
            type=int, help='size to which sequences will be truncated or padded')
    ap.add_argument('--num-workers', required=False, default=4,
            type=int, help='number of workers to use when loading data')
    ap.add_argument('--encode-as-image', required=False, action='store_true')

    ap.add_argument('--data-path', type=str, required=True, help='where the\
                    data is stored, in structure of <data-path>/<test, train, val>')
    ap.add_argument('--lr', type=str, default=1e-3, help='learning rate')

    ap.add_argument('--model-dir', type=str, required=True, help='where to save\
    trained models')
    ap.add_argument('--model-name', type=str, required=True, help='the name of\
            the model you want to train')

    ap.add_argument('--focal-loss', action='store_true', help='whether or not \
            to use focal loss, defined in losses.py')

    args = ap.parse_args()

    data_root = args.data_path
    batch_size = args.batch_size
    max_sequence_length = args.max_sequence_length
    binary_multilabel = args.binary_multilabel
    multiclass = args.multiclass
    num_workers = args.num_workers
    n_epochs = args.epochs
    encode_as_image = args.encode_as_image
    focal_loss = args.focal_loss
    model_name_suffix = args.model_name

    model_dir = args.model_dir


    test = glob(os.path.join(data_root, '*test*'))
    train = glob(os.path.join(data_root, '*train*'))
    valid = glob(os.path.join(data_root, '*val*'))

    test = utils.ProteinSequenceDataset(test,
                              max_sequence_length,
                              encode_as_image,
                              utils.N_CLASSES,
                              binary_multilabel)

    train = utils.ProteinSequenceDataset(train,
                               max_sequence_length,
                               encode_as_image,
                               utils.N_CLASSES,
                               binary_multilabel)

    validation  = utils.ProteinSequenceDataset(valid,
                                     max_sequence_length,
                                     encode_as_image,
                                     utils.N_CLASSES,
                                     binary_multilabel)

    train = torch.utils.data.DataLoader(train, batch_size=batch_size,
            num_workers=num_workers)
    test = torch.utils.data.DataLoader(test, batch_size=batch_size, 
            num_workers=num_workers)
    valid = torch.utils.data.DataLoader(validation, batch_size=batch_size,
            num_workers=num_workers)

    if args.deepfam:
        deepfam_config = {
                'n_classes':utils.N_CLASSES,
                'kernel_size': [8, 12, 16, 20, 24, 28, 32, 36],
                'n_filters': 150,
                'dropout': 0.3,
                'vocab_size': 23,
                'hidden_units': 2000,
                'multilabel_classification': binary_multilabel,
                'lr':1e-3,
                'alphabet_size':len(utils.PROT_ALPHABET),
                'optim':torch.optim.Adam,
                'loss_func':torch.nn.BCEWithLogitsLoss() if not focal_loss else l.FocalLoss()
                }
        model = m.DeepFam(deepfam_config)
        model.train_dataloader = train
        model.val_dataloader = valid
        model_name = 'deepfam{}.h5'
    elif args.deepnog:
        model = m.make_deepnog(n_classes, binary_multilabel)
        model_name = 'deepnog{}.h5'
    elif args.attn:
        model = m.attn_model(max_sequence_length, n_classes, binary_multilabel)
        model_name = 'attn{}.h5'
    else:
        raise ValueError('one of <deepnog, deepfam, attn> required as\
                command-line-arg')

    unique_time = str(int(time.time()))
    model_name = model_name.format(unique_time)

    logdir = os.path.join('logs', unique_time)
    os.makedirs(logdir, exist_ok=True)
    tboard = pl.loggers.tensorboard.TensorBoardLogger(logdir)

    trainer = pl.Trainer(gpus=1, max_epochs=20, logger=tboard)

    trainer.fit(model, train, valid)
    model_name = os.path.join(model_dir, model_name)
    test = train

    with torch.no_grad():
        for batch in test:
            x, y= batch
            preds = model.class_act(model(x))
            gt = (preds >= 0.5).numpy()

            xx = np.round(preds.numpy().ravel())
            yy = y.numpy().ravel()
            print(xx.shape)
            print(yy.shape)
            print(confusion_matrix(xx, yy))

    torch.save(model, model_name)
        # test_stats = model(test)
        # print(test_stats)
