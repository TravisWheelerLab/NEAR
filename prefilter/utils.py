import os
import json
import tensorflow as tf
import numpy as np

import pdb

from glob import glob
from typing import Callable
from random import shuffle, seed

seed(1)

PROT_ALPHABET = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8,
             'K' : 9, 'L' : 10, 'M' : 11, 'N' : 12, 'P' : 13, 'Q' : 14, 'R' : 15, 'S' : 16, 
             'T' : 17, 'V' : 18, 'W' : 19, 'X' : 20, 'Y' : 21, 'Z' : 22 }

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)
SEQUENCES_PER_SHARD = 450000
N_CLASSES = 17646 # number of classes in our dataset as predicted by hmmsearch

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

    if isinstance(label, list):
        feature = {'protein':_int64_feature(protein),
                   'label':_int64_feature(label)}
    else:
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
                prot_seq = sequence.replace('\n', '').upper()
                prot_seq = [PROT_ALPHABET[s] for s in prot_seq]
                tf_example = serialize_example(label, prot_seq)
                writer.write(tf_example)


def read_sequences_from_json(json_file):
    '''
    json_file contains a dictionary of raw AA sequences mapped to their
    associated classes, classified by hmmsearch with an MSA of your choice.
   
    Saves an json file mapping the Pfam accession ID reported by hmmsearch (this
    isn't general, since we're only working with Pfam-trained HMMs available on
    pfam.xfam.org) for easy lookup later on in the classification pipeline. This
    json file is called 'name-to-label.json'.

    Returns a list of lists. Each list contains the raw AA sequence as its
    second element and the list of hmmsearch determined labels as its first
    (there can be more than one if hmmsearch returns multiple good matches for
    an AA sequence).

    These lists will be passed to a subroutine that shards them into multiple
    tfrecord files on disk. Shuffling is done once to the whole dataset (right
    before the return call in this function), and when data are being sent to
    the model via tensorflow's shuffle function.
    '''

    with open(json_file, 'r') as f:
        sequence_to_label = json.load(f)

    name_to_integer_label = {}
    integer_label = 0

    fout = os.path.join(os.path.dirname(json_file), 'name-to-label.json')

    if os.path.isfile(fout):
        with open(fout, 'r') as f:
            name_to_integer_label  = json.load(f)
        integer_label = max(name_to_integer_label.values())

    len_before = len(name_to_integer_label)

    for seq in sequence_to_label.keys():
        for label in sequence_to_label[seq]:
            if label not in name_to_integer_label:
                name_to_integer_label[label] = integer_label
                integer_label += 1

    if len_before != len(name_to_integer_label):
        s = 'saving new labels, number of unique classes went from {} to {}'.format(len_before,
                len(name_to_integer_label))
        print(s)
        with open(fout, 'w') as f:
            # overwrite old file if it exists
            json.dump(name_to_integer_label, f)

    sequence_to_integer_label = []

    for sequence, labels in sequence_to_label.items():
        sequence_to_integer_label.append([[name_to_integer_label[l]
            for l in labels], sequence])


    shuffle(sequence_to_integer_label)
    return sequence_to_integer_label


def read_sequences_from_fasta(files, save_name_to_label=False):
    ''' 
    returns list of all sequences in the fasta files. 
    list is a list of two-element lists with the first element the class
    label and the second the protein sequence.

    Assumes that each fasta file has the class name of the proteins as its
    filename. TODO: implement logic that parses the fasta header to get a class
    name.

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
        max_sequence_length,
        encode_as_image,
        multiple_labels,
        softmax):

    '''
    I need to have tensors of uniform size. I need to implement
    logic that pads or cuts off AA sequences according to sequence_length.
    '''

    if multiple_labels:
        feature_description = {
            'protein': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
                allow_missing=True),
            'label': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
                allow_missing=True),
        }
    else:
        feature_description = {
            'protein': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,
                allow_missing=True),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }

    def parse_tfrecord(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    def encode_protein(inputs):
        protein = inputs.get('protein')
        label = inputs.get('label')

        if tf.greater(tf.shape(protein), max_sequence_length):
            protein = tf.slice(protein, begin=[0], size=[max_sequence_length])
        if encode_as_image:
            oh = tf.expand_dims(tf.one_hot(protein, depth=LEN_PROTEIN_ALPHABET),
                    -1) # one hot
        else:
            oh = protein

        if multiple_labels:
            label_oh = tf.reduce_max(tf.one_hot(label, depth=N_CLASSES), axis=0)

<<<<<<< HEAD
        elif multiple_labels and not softmax:
            label_oh = tf.reduce_max(tf.one_hot(label, depth=N_CLASSES), axis=0)
            label_oh = label_oh / tf.cast(tf.math.count_nonzero(label_oh), tf.float32)
        else:
            pass
=======
            # label = tf.reduce_sum(tf.one_hot(label, depth=N_CLASSES), axis=0)
            label = tf.reduce_max(tf.one_hot(label, depth=N_CLASSES), axis=0)
>>>>>>> 0a8e8e6c8eb4e0810ee1166450cb46b046c5d278

        return oh, label


    files = tf.io.gfile.glob(tfrecord_path)
    files = tf.random.shuffle(files)
    shards = tf.data.Dataset.from_tensor_slices(files)

    dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(map_func=parse_tfrecord,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=encode_protein,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if multiple_labels:
        pad_label_kwarg = [None]
    else:
        pad_label_kwarg = []

    if encode_as_image:
        dataset = dataset.padded_batch(batch_size,
                padded_shapes=([max_sequence_length, 23, 1], pad_label_kwarg))
    else:
        dataset = dataset.padded_batch(batch_size,
                padded_shapes=([max_sequence_length], pad_label_kwarg))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    json_root = '../data/clustered-shuffle/json/'
    data_root = '../data/clustered-shuffle/'

    # test = read_sequences_from_json(json_root + 'test-sequences-and-labels.json')
    # train = read_sequences_from_json(json_root + 'train-sequences-and-labels.json')
    # validation = read_sequences_from_json(json_root + 'val-sequences-and-labels.json')

    # out_template = '/data-{}-of-{}.tfrecord'

    # shard_sequences(data_root + 'train' +  out_template, train)
    # shard_sequences(data_root + 'test' +  out_template, test)
    # shard_sequences(data_root + 'validation' +  out_template, validation)

    test = data_root + 'test/*'
    train = data_root + 'train/*'
    validation = data_root + 'validation/*'

<<<<<<< HEAD
    test = make_dataset(test, 1, 1000, 1024, encode_as_image=False, 
            multiple_labels=False)
    train = make_dataset(train, 1, 1000, 1024, encode_as_image=False, 
            multiple_labels=False)
    validation = make_dataset(validation, 1, 1000, 1024, encode_as_image=False, 
            multiple_labels=False)
=======
    batch_size = 16
    test = make_dataset(test, batch_size, 1000, 1024, encode_as_image=False, 
            multiple_labels=True)
    train = make_dataset(train, batch_size, 1000, 1024, encode_as_image=False, 
            multiple_labels=True)
    validation = make_dataset(validation, batch_size, 1000, 1024, encode_as_image=False, 
            multiple_labels=True)
>>>>>>> 0a8e8e6c8eb4e0810ee1166450cb46b046c5d278

    model_path = '../models/alienware_deepnog.h5'

    def sched(lr):
        return lr

    wr = WarmUp(0.01, sched, 1000)
    # model = tf.keras.models.load_model(model_path, custom_objects={'WarmUp': wr})

    # print(model)

    tot = 0
    i = 0
    unique = set()
    for feat, lab in train:
        unique.update(np.unique(lab))
    print(unique)
