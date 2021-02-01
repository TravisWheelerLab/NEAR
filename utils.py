import os
import re
import numpy as np
from glob import glob
from random import shuffle, seed
import matplotlib.pyplot as plt
import tensorflow as tf

seed(1)

PROT_ALPHABET = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8,
             'K' : 9, 'L' : 10, 'M' : 11, 'N' : 12, 'P' : 13, 'Q' : 14, 'R' : 15, 'S' : 16, 
             'T' : 17, 'V' : 18, 'W' : 19, 'X' : 20, 'Y' : 21, 'Z' : 22 }

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)

def _read_fasta(fasta, label):

    with open(fasta, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    current_seq = ''
    first_line = True
    seqs = []

    for line in lines:
        if first_line:
            first_line = False
        else:
            if re.search(">", line) and current_seq != "":
                current_seq = current_seq[1:]
                if (len(current_seq) >= 31):
                        seqs.append([current_seq, label])
                current_seq = ""
            else:
                current_seq += line
    return seqs


def read_sequences(fasta_files):
    seqs = []
    # since individual fasta files store a single family,
    # class code comes from the name of the fasta file
    class_code = -1 # this needs to be canonized
    # most likely gonna need to pad data here
    mxlen = 0

    for fasta in set(fasta_files):

        if fasta.endswith('neg'):
            seqs.append(_read_fasta(fasta, label=-1))
        else:
            class_code += 1
            seq = _read_fasta(fasta, label=class_code)
            lengths = [len(s[0]) for s in seq]
            if len(lengths):
                maxlen = max(lengths)
                if maxlen > mxlen:
                    mxlen = maxlen

            seqs.append(seq)
    

    return seqs, mxlen


def encode_protein_as_one_hot_vector(protein, maxlen=None):
    # input: raw protein string of arbitrary length 
    # output: np.array() of size (1, maxlen, length_protein_alphabet)
    # Each row in the array is a separate character, encoded as 
    # a one-hot vector

    protein = protein.upper()
    one_hot_encoding = np.zeros((1, maxlen, LEN_PROTEIN_ALPHABET))

    # not the label, the actual encoding of the protein.
    # right now, it's a stack of vectors that are one-hot encoded with the character of the
    # alphabet
    # Could this be vectorized? Yeah. But POITROAE
    for i, residue in enumerate(protein):

        if i > maxlen-1:
            break
        
        one_hot_encoding[0, i, PROT_ALPHABET[residue]] = 1

    return one_hot_encoding


def encode_protein_as_number(protein, maxlen=None):
    # input: raw protein string of arbitrary length 
    # output: np.array() of size (1, maxlen, 1)
    # Each row in the array is a separate character, encoded as 
    # a number. This means embeddings (specifically keras.layers.Embedding) 
    # are easy to compute.

    protein = protein.upper()
    encoding = np.zeros((1, maxlen, 1))

    # not the label, the actual encoding of the protein.
    # right now, it's a stack of vectors that are one-hot encoded with the character of the
    # alphabet
    # Could this be vectorized? Yeah. But POITROAE
    for i, residue in enumerate(protein):

        if i > maxlen-1:
            break
        
        encoding[0, i] = PROT_ALPHABET[residue]

    return encoding


def encode_sequences(seqs, encoding_func, subsample=None, maxlen=None):

    # there are 0-length sequences b/c the read_sequences code
    # throws out sequences that are less than 31 residues long
    twoD = False
    if maxlen and subsample and 'vector' not in encoding_func.__name__:
        features = np.zeros((len(seqs)*subsample, maxlen, 1))
        targets = np.zeros((len(seqs)*subsample))

    elif (maxlen and subsample) and 'vector' in encoding_func.__name__:
        features = np.zeros((len(seqs)*subsample, maxlen, LEN_PROTEIN_ALPHABET))
        targets = np.zeros((len(seqs)*subsample))
        twoD = True
    else:
        features = []
        targets = []

    # this can be trivially parallelized
    cnt = 0
    cutoff = 0
    neg_count = 0
    for i, family in enumerate(seqs):

        # family is a list of nested lists. each nested list contains
        # the protein sequence and the class label

        shuffle(family)
        if subsample:
            family = family[:subsample]

            if len(family) < subsample:
                cutoff += subsample - len(family)

        for seq in family:

            # we just chop the end of the protein sequence here
            # but we could just as easily throw out sequences that
            # don't pass some length threshold

            encoding = encoding_func(seq[0], maxlen=maxlen)

            if maxlen and subsample:
                features[cnt] = encoding
                targets[cnt] = int(seq[1])
                cnt += 1
                
            else:
                features.append(encoding)
                targets.append(int(seq[1]))

    if cutoff != 0:
        # ... interestingly 
        features = features[:-cutoff]
        targets = targets[:-cutoff]

    if np.any(targets == -1):
        targets[targets == -1] = np.max(targets) + 1

    n_classes = np.max(targets)
    shuffle_indices = np.random.choice(np.arange(features.shape[0]), size=features.shape[0],
            replace=False)

    # should mix nicely between all classes. Probably going
    # to want a way to balance classes artificially
    
    if twoD:
        features = np.expand_dims(features[shuffle_indices], -1)
        # features = np.transpose(features, (0, 2, 1, 3))
        targets  = np.expand_dims(targets[shuffle_indices], -1)
    else:
        features = features[shuffle_indices]
        targets = targets[shuffle_indices]

    return features, targets, n_classes


def train_test_val_split(features, targets, splits=(90, 5, 5), batch_size=1,
        shuffle_len=100):
    # splits should be integers b/t 0 and 100

    idx = np.random.choice(np.arange(features.shape[0]), size=features.shape[0],
            replace=False)
    assert(splits[0] > 1)
    split_train = int(idx.shape[0]*splits[0]/100)
    split_test = int(idx.shape[0]*splits[1]/100)
    idx_train = idx[:split_train]
    idx_test = idx[split_train:split_train+split_test]
    idx_val = idx[-split_test:]

    # returns test,train,val
    train_dset = _create_dataset(features[idx_train], targets[idx_train],
            batch_size=batch_size, shuffle_len=shuffle_len, val=False)
    test_dset = _create_dataset(features[idx_test], targets[idx_test],
            batch_size=batch_size, shuffle_len=shuffle_len, val=True) 
    # shuffling test/val doesn't matter
    val_dset = _create_dataset(features[idx_val], targets[idx_val],
            batch_size=batch_size, shuffle_len=shuffle_len, val=True)

    return train_dset, test_dset, val_dset


def _create_dataset(features, targets, batch_size, shuffle_len, val=False):

    features = tf.data.Dataset.from_tensor_slices(features)
    targets = tf.data.Dataset.from_tensor_slices(targets)
    zipped = tf.data.Dataset.zip((features, targets))
    if val:
        return zipped.batch(batch_size).shuffle(shuffle_len)
    else:
        return zipped.batch(batch_size).shuffle(shuffle_len).repeat()



if __name__ == '__main__':

    files = sorted(glob('./fasta/*'))
    sequences, maxlen = read_sequences(files)
    maxlen = 500
    encoding_func = encode_protein_as_number
    features, targets, n_classes = encode_sequences(sequences, encoding_func,
            subsample=30, maxlen=maxlen)
    print(features.shape)
    batch_size = 32
    # train_test_val_split(features, targets, batch_size=batch_size)
