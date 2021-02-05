import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
import pdb
import sys
import pickle
import random
from random import shuffle
from glob import glob
from functools import partial, update_wrapper

random.seed(1)

from sklearn.metrics import confusion_matrix
import utils as utils
import models as m

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.callbacks as cbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


# ABCDEFGHIKLMNPQRSTVWXYZ
# Class code comes from the filename, which stores protein sequences
# for a given family
# It seems like only 1000 sequences from one directory of the fasta files available are
# used to train and evaluate the model, which come from 172 families (173 with the 'negative'
# family included. 
# I don't know where N=858 came from since the softmax is only 173-dimensional.
# The results showed a top-90 accuracy of 100% for the positive classes and no entries 
# in the top-100 classes for randomly shuffled data.

# Theoretically, HMMs should encode some relevant information about a family, right?

# Use an HMM in a generative fashion to make more training data that is slightly different than
# every known sequence but close enough to fool the model. 

# Distances between residues, which are important for predicting structure 
# (and therefore function), are not necessarily proportional to the distance between the symbol
# we use to describe a residue in any given sequence.

# So maybe a model that DOES incorporate very long-range context (i.e. the whole length of the 
# protein string) would be necessary.
# Possibly stacks of CNN layers?

# To do here:
# Batch the data for efficiency's sake (throw out the last or first characters)
# Make a training/development/validation pipeline
# Figure out various ways to encode protein sequences (BPE, k-mers, one-hot?)

def model2d(n_classes):
    model = Sequential()
    model.add(Conv2D(75, (31, 23), input_shape=(None, None, 1), activation='relu', use_bias=True))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(50, activation='relu', use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax', name='output'))
    return model


def model2d2(n_classes):
    model = Sequential()
    model.add(Conv2D(75, (23, 31), input_shape=(None, None, 1), activation='relu',
        padding='same'))
    model.add(Conv2D(75, (23, 31), activation='relu',
        padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(50, activation='relu', use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax', name='output'))
    return model


def model(n_classes):
    model = Sequential()
    model.add(Embedding(utils.LEN_PROTEIN_ALPHABET, 128))
    model.add(Conv1D(32, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(Conv1D(256, 3, activation='relu', padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(256, activation='relu', use_bias=True))
    model.add(Dense(128, activation='relu', use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax', name='output'))
    return model

def oa(cmat):
    total = np.sum(np.sum(cmat, axis=1))
    correct = np.sum(np.diag(cmat))
    return correct / total

if __name__ == '__main__':

    train = './data/train/*'
    test = './data/test/*'
    valid= './data/validation/*'

    batch_size = 64
    shuffle_buffer = 1000
    max_sequence_length = 256
    encode_as_image = False

    train = utils.make_dataset(train,
            batch_size, shuffle_buffer, max_sequence_length, encode_as_image)
    train = train.repeat()
    test = utils.make_dataset(test,
            batch_size, shuffle_buffer, max_sequence_length, encode_as_image)
    validation  = utils.make_dataset(valid,
            batch_size, shuffle_buffer, max_sequence_length, encode_as_image)

    # top5 accuracy.
    tpk_metric = keras.metrics.sparse_top_k_categorical_accuracy
    k = 90
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

    n_classes = 858
    lr_schedule = utils.WarmUp(initial_learning_rate,
            lr_schedule, 3000)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    #model = m.attn_model(max_sequence_length, n_classes)
    model = m.make_deepnog(n_classes)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=['accuracy', tpk])
    model.summary()
    tb = cbacks.TensorBoard(log_dir='./logs/')

    model.fit(train,
              steps_per_epoch=680000//batch_size,
              epochs=100,
              validation_data=validation,
              callbacks=[tb],
              verbose=1)

    test_stats = model.evaluate(test)
    print(test_stats)

