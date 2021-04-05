import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
import time
import pdb
import sys
import pickle
import random

random.seed(1)

import utils as utils
import models as m

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as cbacks

from random import shuffle
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
    ap.add_argument('--shuffle-buffer-size', required=False, default=1000,
            type=int, help='num examples in the shuffle buffer, larger means\
            more complete in-memory shuffling')
    ap.add_argument('--max-sequence-length', required=False, default=256,
            type=int, help='size to which sequences will be truncated or padded')

    ap.add_argument('--data-path', type=str, required=True, help='where the\
                    data is stored, in structure of <data-path>/<test, train, val>'
    ap.add_argument('--model-dir', type=str, required=True, help='where to save\
    trained models')

    args = ap.parse_args()

    data_root = ap.data_path
    batch_size = args.batch_size
    shuffle_buffer = args.shuffle_buffer_size
    max_sequence_length = args.max_sequence_length
    binary_multilabel = args.binary_multilabel
    multiclass = args.multiclass
    n_epochs = args.epochs

    model_dir = args.model_dir

    encode_as_image = False

    test = os.path.join(data_root, 'test/*')
    train = os.path.join(data_root, 'train/*')
    valid = os.path.join(data_root, 'validation/*')

    train = utils.make_dataset(train,
                               batch_size,
                               shuffle_buffer,
                               max_sequence_length,
                               encode_as_image,
                               binary_multilabel,
                               multiclass)
    train = train.repeat()
    test = utils.make_dataset(test,
                              batch_size,
                              shuffle_buffer,
                              max_sequence_length,
                              encode_as_image,
                              binary_multilabel,
                              multiclass)

    validation  = utils.make_dataset(valid,
                                     batch_size,
                                     shuffle_buffer,
                                     max_sequence_length,
                                     encode_as_image,
                                     binary_multilabel,
                                     multiclass)

    # top-k accuracy.
    tpk_metric = keras.metrics.sparse_top_k_categorical_accuracy
    k = 5
    tpk = partial(tpk_metric, k=k)
    # for making the terminal output shorter.
    tpk = update_wrapper(tpk, tpk_metric)
    tpk.__name__ = 'tp{}'.format(k)

    # model/optimizer setup

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    if args.deepfam:
        model = m.make_deepfam(n_classes, binary_multilabel)
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

    if binary_multilabel:
        model.compile(loss='binary_crossentropy', optimizer=opt,
                metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                metrics=['accuracy', tpk])

    model.summary()

    logdir = os.path.join('logs', unique_time)
    os.makedirs(logdir, exist_ok=True)

    tb = cbacks.TensorBoard(log_dir=logdir)

    model.fit(train,
              steps_per_epoch=420000 // batch_size,
              epochs=n_epochs,
              validation_data=validation,
              callbacks=[tb],
              verbose=1)

    model_name = os.path.join(model_dir, model_name)

    model.save(model_name)

    test_stats = model.evaluate(test)
    print(test_stats)
