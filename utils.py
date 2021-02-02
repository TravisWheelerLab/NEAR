import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pdb

from glob import glob
from random import shuffle, seed
from collections import namedtuple
from functools import partial

seed(1)

PROT_ALPHABET = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8,
             'K' : 9, 'L' : 10, 'M' : 11, 'N' : 12, 'P' : 13, 'Q' : 14, 'R' : 15, 'S' : 16, 
             'T' : 17, 'V' : 18, 'W' : 19, 'X' : 20, 'Y' : 21, 'Z' : 22 }

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)
SEQUENCES_PER_SHARD = 450000

def read_fasta(fasta, label):

    def _parse_line(line):
        idx = line.find('\n')
        family = line[:idx]
        sequence = line[idx+1:].replace('\n', '')
        return [label, sequence]

    with open(fasta, 'r') as f:
        names_and_sequences = f.read()
        lines = names_and_sequences.split('>')
        sequences = list(map(_parse_line, lines))

    return sequences

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

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(label, protein):
    '''
    The bytes feature here could definitely be encoded as a 4 bit integer
    '''

    feature = {'protein':_int64_feature(protein),
               'label':_int64_feature([label])}

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()


def shard_sequences(record_file_template, sequences):
    '''
    record_file_template looks something like this: sequences.{}-of-{}.tfrecord'
    '''
    shuffle(sequences)
    n_shards = np.ceil(len(sequences) / SEQUENCES_PER_SHARD)
    for j, i in enumerate(range(0, len(sequences), SEQUENCES_PER_SHARD)):
        sharded_sequences = sequences[i:i+SEQUENCES_PER_SHARD]
        record_file = record_file_template.format(j, n_shards)
        with tf.io.TFRecordWriter(record_file) as writer:
            for i, (label, sequence) in enumerate(sharded_sequences):
                tf_example = serialize_example(label,
                        [PROT_ALPHABET[s] for s in sequence.upper()])

                writer.write(tf_example)


def read_sequences_from_fasta(files, save_name_to_label=False):
    ''' 
    returns list of all sequences in the fasta files. 
    list is a list of two-element lists with the first element the class
    label and the second the protein sequence.

    does not take care of train/dev/val splits. This is trivially
    implemented.

    '''
    name_to_label = {}
    cnt = 0
    sequences = []

    for f in files:
        seq = read_fasta(f, cnt)
        name_to_label[os.path.basename(f)] = cnt
        cnt += 1
        sequences.extend(filter(lambda x: len(x[1]), seq))

    if save_name_to_label:
        with open('name_to_class_code.json', 'w') as f:
            json.dump(name_to_label, f)

    shuffle(sequences)
    return sequences


def ttv_split(sequences):
    '''

    all of these functions rely on all of the sequences fitting in memory for
    complete shuffling. Implementing out-of-memory shuffling of potentially
    hundreds of Gb of fasta files is trivially doable (requires some
    bookkeeping), but seeing as ~1M sequences fit as strings in a nested list in
    240M, this won't be a problem on pretty much any server or even a powerful
    laptop.
    '''

    shuffle(sequences)
    train = sequences[:int(0.8*len(sequences))]
    test = sequences[int(0.8*len(sequences)):]
    valid = test[:len(test)//2]
    test = test[len(test)//2:]

    return train, test, valid


def make_dataset(tfrecord_path,
        batch_size,
        buffer_size,
        max_sequence_length): 

    '''
    I need to have tensors of uniform size. I need to implement
    logic that pads or cuts off AA sequences according to sequence_length.
    Is this going to be a problem for generalization? 
    Will byte-pair encoding be the most useful thing to do here?
    '''

    feature_description = {
        'protein': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
            allow_missing=True),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def parse_tfrecord(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    def encode_protein(inputs):
        protein = inputs.get('protein')
        if tf.greater(tf.shape(protein), max_sequence_length):
            protein = tf.slice(protein, begin=[0], size=[max_sequence_length])
        return tf.expand_dims(tf.one_hot(protein, depth=LEN_PROTEIN_ALPHABET),
                -1), inputs.get('label')


    files = tf.io.gfile.glob(tfrecord_path)
    files = tf.random.shuffle(files)
    shards = tf.data.Dataset.from_tensor_slices(files)

    dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(map_func=parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=encode_protein, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(batch_size,
            padded_shapes=([max_sequence_length, 23, 1], []))
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset



if __name__ == '__main__':

    files = sorted(glob('./fasta/*'))

    # sequences = read_sequences_from_fasta(files)
    # train, test, valid = ttv_split(sequences)

    # shard_sequences('./data/train/data-{}-of-{}.tfrecord', train)
    # shard_sequences('./data/test/data-{}-of-{}.tfrecord', test)
    # shard_sequences('./data/validation/data-{}-of-{}.tfrecord', valid)

    train = './data/train/*'
    test = './data/test/*'
    valid = './data/valid/*'

    train = make_dataset(train, 64, 1000, 256)

    for i, feat in enumerate(train):
        print(feat[0].shape, feat[1].shape)
        exit()
