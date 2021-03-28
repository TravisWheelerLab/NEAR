import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
import pdb
import sys
import pickle
import random
from random import shuffle
from glob import glob
from functools import partial, update_wrapper

random.seed(1)

import utils as utils
import models as m

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as cbacks

if __name__ == '__main__':

    data_root = '../data/clustered-shuffle/'
    test = data_root + 'test/*'
    train = data_root + 'train/*'
    valid = data_root + 'validation/*'

    batch_size = 64
    shuffle_buffer = 1000
    max_sequence_length = 256
    encode_as_image = False
    multilabel = True

    train = utils.make_dataset(train,
                               batch_size,
                               shuffle_buffer,
                               max_sequence_length,
                               encode_as_image,
                               multilabel)
    train = train.repeat()
    test = utils.make_dataset(test,
                              batch_size,
                              shuffle_buffer,
                              max_sequence_length,
                              encode_as_image,
                              multilabel)

    validation  = utils.make_dataset(valid,
                                     batch_size,
                                     shuffle_buffer,
                                     max_sequence_length,
                                     encode_as_image,
                                     multilabel)

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

    n_classes = 17646 # not true... need to change
    lr_schedule = utils.WarmUp(initial_learning_rate,
            lr_schedule, 3000)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    # model = m.attn_model(max_sequence_length, n_classes, multilabel)
    # model = m.make_deepnog(n_classes, multilabel)

    model = m.make_deepfam(n_classes, multilabel)

    if multilabel:
        model.compile(loss='binary_crossentropy', optimizer=opt,
                metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                metrics=['accuracy', tpk])
    model.summary()
    tb = cbacks.TensorBoard(log_dir='./logs/')

    model.fit(train,
              steps_per_epoch=420141 // batch_size,
              epochs=100,
              validation_data=validation,
              callbacks=[tb],
              verbose=1)

    test_stats = model.evaluate(test)
    print(test_stats)
