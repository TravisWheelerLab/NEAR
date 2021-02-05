import os
import re
import pdb
import sys
import pickle
import random
from random import shuffle

random.seed(1)

from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv1D

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

def model():
    model = Sequential()
    model.add(Conv2D(75, (23, 31), input_shape=(None, None, 1), activation='relu', use_bias=True))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(50, activation='relu', use_bias=True))
    model.add(Dropout(0.2))
    model.add(Dense(173, activation='softmax', name='output'))
    return model

def train(train_set, test_set, class_cnt, out_path=None):

    model = model()
    opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    num_epochs = 3
    train_cnt = len(train_set)
    val_cnt = len(test_set)

    for e in range (num_epochs):
            all_loss = []
            all_acc = []
            i = 1
            total_loss = 0
            total_acc = 0
            # unison_shuffle(train_x, train_y)
            random.shuffle(train_set)
    
            train_correct = 0

            for ex in train_set:
                    x, y = one_hot_encode(ex, class_cnt)
                    (loss, acc) = model.train_on_batch(x, y)
                    train_correct += acc

            print()
            total_val_loss = 0
            total_val_acc = 0

            for ex in test_set:
                    x, y = one_hot_encode(ex, class_cnt)
                    (loss, acc) = model.test_on_batch(x, y)
                    total_val_loss += loss
                    total_val_acc += acc

            val_loss = total_val_loss /  val_cnt
            val_acc = total_val_acc / val_cnt

            print('Epoch: {}  train acc: {:3.4f}, dev acc: {:3.4f}'.format(e, 
                train_correct/len(train_set), val_acc))


    return model


def validate_final(model, test_set, class_cnt, show_time=False, save=False,
        n=None):
    predictions = []
    labels = []
    preds = []

    for ex in test_set:
            x, y = one_hot_encode(ex, class_cnt)
            pred = model.predict(
                    x         =x,
                    batch_size=None,
                    verbose   =0,
                    steps     =None
            ).flatten()     
                    
            top_preds = np.argsort(pred)[-100:][::-1]
            predictions.append(top_preds)
            preds.append(pred.argmax())


            l = y.argmax()
            labels.append(l)

    v_size = len(test_set)

    if save:
        model.save('data_{}.h5'.format(n))


    top_1 = 0
    top_3 = 0
    top_5 = 0
    top_10 = 0
    top_20 = 0
    top_30 = 0
    top_40 = 0
    top_50 = 0
    top_60 = 0
    top_70 = 0
    top_80 = 0
    top_90 = 0
    top_100 = 0

    for p, l in zip(predictions, labels):
            if (l in p[:1]):
                    top_1 += 1

            if (l in p[:3]):
                    top_3 += 1

            if (l in p[:5]):
                    top_5 += 1

            if (l in p[:10]):
                    top_10 += 1

            if (l in p[:20]):
                    top_20 += 1

            if (l in p[:30]):
                    top_30 += 1
            
            if (l in p[:40]):
                    top_40 += 1

            if (l in p[:50]):
                    top_50 += 1

            if (l in p[:60]):
                    top_60 += 1

            if (l in p[:70]):
                    top_70 += 1

            if (l in p[:80]):
                    top_80 += 1

            if (l in p[:90]):
                    top_90 += 1

            if (l in p[:100]):
                    top_100 += 1

    print("Top 1\t{:3.4f}\n".format(top_1/v_size));
    print("Top 3\t{:3.4f}\n".format(top_3/v_size));
    print("Top 5\t{:3.4f}\n".format(top_5/v_size));
    print("Top 10\t{:3.4f}\n".format(top_10/v_size));
    print("Top 20\t{:3.4f}\n".format(top_20/v_size));
    print("Top 30\t{:3.4f}\n".format(top_30/v_size));
    print("Top 40\t{:3.4f}\n".format(top_40/v_size));
    print("Top 50\t{:3.4f}\n".format(top_50/v_size));
    print("Top 60\t{:3.4f}\n".format(top_60/v_size));
    print("Top 70\t{:3.4f}\n".format(top_70/v_size));
    print("Top 80\t{:3.4f}\n".format(top_80/v_size));
    print("Top 90\t{:3.4f}\n".format(top_90/v_size));
    print("Top 100\t{:3.4f}\n".format(top_100/v_size));

    return confusion_matrix(labels, preds), labels, preds

def save_model(model, path):
    model_json = model.to_json()

    json_path = path + "/model.json"
    summary_path = path + "/model_summary.txt"

    with open(json_path, "w") as json_file:
        json_file.write(model_json)

def main():
    train_negatives = []
    train_positives = []
    test_negatives = []
    test_positives = []
    class_cnts = []

    
    for i in range(1, 6):
            neg, cnt = read_sequences("fasta/{}neg/".format(i), 'neg')
            class_cnts.append(cnt)
            random.shuffle(neg)
            s = int(0.8 * len(neg))
            train_negatives.append(neg[:s])
            test_negatives.append(neg[s:])

    for i in range(1,6):
            pos, cnt = read_sequences("fasta/{}pos/".format(i), 'pos')
            random.shuffle(pos)
            s = int(0.8 * len(pos))
            train_positives.append(pos[:s])
            test_positives.append(pos[s:])

    train_set = (train_positives[0] 
                            + train_negatives[0]
                            + train_negatives[1]
                            + train_negatives[2]
                            + train_negatives[3]
                            + train_negatives[4]
    )

    neg_shuffle = train_negatives[0] + train_negatives[1] + train_negatives[2] +\
            train_negatives[3] + train_negatives[4] 

    pos_shuffle = train_positives[0] + train_positives[1] + train_positives[2] +\
            train_positives[3] + train_positives[4] 

    idx_neg = np.random.choice(np.arange(len(neg_shuffle)), size=len(neg_shuffle)//30)
    idx_pos = np.random.choice(np.arange(len(pos_shuffle)), size=len(pos_shuffle))
    
    train_neg = np.asarray(neg_shuffle)[idx_neg]
    train_pos = np.asarray(pos_shuffle)[idx_pos]

    # print(np.unique(train_neg[:, 1]))
    # print(np.unique(train_pos[:, 1]))
    # print(train_pos.shape)
    # print(train_neg.shape)

    train_set_bal = np.concatenate((train_pos, train_neg), axis=0)

    idx = np.random.choice(np.arange(train_set_bal.shape[0]), size=train_set_bal.shape[0])
    train_set_bal = train_set_bal[idx]

    test_set = (test_positives[0] 
                            + test_negatives[0]
                            + test_negatives[1]
                            + test_negatives[2]
                            + test_negatives[3]
                            + test_negatives[4]
    )


    neg_shuffle = test_negatives[0] + test_negatives[1] + test_negatives[2] +\
            test_negatives[3] + test_negatives[4] 

    pos_shuffle = test_positives[0] + test_positives[1] + test_positives[2] +\
            test_positives[3] + test_positives[4] 

    idx_neg = np.random.choice(np.arange(len(neg_shuffle)), size=len(neg_shuffle)//30)
    idx_pos = np.random.choice(np.arange(len(pos_shuffle)), size=len(pos_shuffle))

    test_neg = np.asarray(neg_shuffle)[idx_neg]
    test_pos = np.asarray(pos_shuffle)[idx_pos]

    # print(np.unique(test_neg[:, 1]))
    # print(np.unique(test_pos[:, 1]))

    # print(test_pos.shape)
    # print(test_neg.shape)

    test_set_bal = np.concatenate((test_pos, test_neg), axis=0)
    idx = np.random.choice(np.arange(train_set_bal.shape[0]), size=train_set_bal.shape[0])
    train_set_bal = train_set_bal[idx]

    
    save = False
    n = len(train_set)

    model = train(train_set_bal, test_set_bal, class_cnts[0])


    model.save('balanced_train_test.h5')

    cmat, labels, preds = validate_final(model, train_set_bal, class_cnts[0],
            save=save, n=n)

    with open('cmat_train.pkl', 'wb') as f:
        pickle.dump(cmat, f)

    cmat, labels, preds = validate_final(model, test_set_bal, class_cnts[0],
            save=save, n=n)

    with open('cmat_test.pkl', 'wb') as f:
        pickle.dump(cmat, f)

    print(np.unique(labels, return_counts=True))
    print(np.unique(preds, return_counts=True))

    # print("Negative same family")
    # validate_final(model, test_negative1[:n], class_cnts[0])
    # print("Negative different family")
    # validate_final(model, test_negative2[:n], class_cnts[0])

def oa(cmat):

    total = np.sum(np.sum(cmat, axis=1))
    correct = np.sum(np.diag(cmat))
    return correct / total

if __name__ == '__main__':
    # main()
    # read_sequences('./fasta/1pos/40S_SA_C')


    with open('./cmat_train.pkl', 'rb') as f:
        cmat_train = pickle.load(f)

    with open('./cmat_test.pkl', 'rb') as f:
        cmat_test = pickle.load(f)

    # plt.imshow(cmat_train[:-1,:-1])
    # plt.show()
    # plt.imshow(cmat_test[:-1,:-1])
    # plt.show()

    print(oa(cmat_train))
    print(oa(cmat_test))
