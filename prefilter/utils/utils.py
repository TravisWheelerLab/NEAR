import os
import json
import torch
import numpy as np

import pdb
import time

from glob import glob
from typing import Callable
from random import shuffle, seed
from collections import defaultdict


seed(1)

__all__ = ['ProteinSequenceDataset', 'get_n_classes']

PROT_ALPHABET = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8,
             'K' : 9, 'L' : 10, 'M' : 11, 'N' : 12, 'P' : 13, 'Q' : 14, 'R' : 15, 'S' : 16, 
             'T' : 17, 'V' : 18, 'W' : 19, 'X' : 20, 'Y' : 21, 'Z' : 22 }

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)

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

    protein = protein.upper().replace('\n', '')
    one_hot_encoding = np.zeros((1, maxlen, LEN_PROTEIN_ALPHABET))

    # not the label, the actual encoding of the protein.
    # right now, it's a stack of vectors that are one-hot encoded with the character of the
    # alphabet
    # Could this be vectorized? Yeah.
    for i, residue in enumerate(protein):

        if i > maxlen-1:
            break
        
        one_hot_encoding[0, i, PROT_ALPHABET[residue]] = 1

    return one_hot_encoding


def read_sequences_from_json(json_file, fout=None):
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
    '''

    with open(json_file, 'r') as f:
        sequence_to_label = json.load(f)

    name_to_integer_label = {}
    integer_label = 0

    if fout is None:
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


class Word2VecStyleDataset(torch.utils.data.Dataset):
    
    def __init__(self,
            json_files,
            max_sequence_length,
            name_to_label_mapping,
            n_negative_samples=5,
            n_classes=None
            ):

        self.max_sequence_length = max_sequence_length
        self.name_to_label_mapping = name_to_label_mapping
        self.n_negative_samples = n_negative_samples

        if n_classes is None:
            self.n_classes = get_n_classes(self.name_to_label_mapping)
        else:
            self.n_classes = n_classes

        self._build_dataset(json_files)

    def _encoding_func(self, x):
        oh = encode_protein_as_one_hot_vector(x, self.max_sequence_length)
        oh = np.argmax(oh, axis=-1)
        return oh.squeeze()

    def _sample_example(self):

        idx = int(np.random.rand()*len(self.sequences))
        target_sequence = self.sequences[idx]
        # grab a random sequence
        x = self.sequences_and_labels[target_sequence] #... and all of the
        # labels that come along with it (pfam ids)

        target_family = np.random.choice(x, size=1)[0]
        y = self.labels_and_sequences[target_family]
        context_sequence = np.random.choice(y, size=1)[0]

        # k, now sample self.n_negative_samples
        negative_examples = []
        i = 0
        while len(negative_examples) < self.n_negative_samples:
            negative_examples = np.random.choice(self.pfam_names, size=self.n_negative_samples)
            i +=1 
            if target_family in set(negative_examples):
                negative_examples = []

        negatives = []
        for negative in negative_examples:
            negatives.append(np.random.choice(self.labels_and_sequences[negative], size=1)[0])

        target = torch.tensor(self._encoding_func(target_sequence))
        context = torch.tensor(self._encoding_func(context_sequence))
        negatives = torch.tensor([self._encoding_func(x) for x in negatives])
        return target, context, negatives


    def _build_dataset(self, json_file):

        with open(json_file, 'r') as src:
            self.sequences_and_labels = json.load(src)

        self.labels_and_sequences = defaultdict(list)
        for prot_seq, accession_ids in self.sequences_and_labels.items():
            for i in accession_ids:
                self.labels_and_sequences[i].append(prot_seq)

        self.sequences = list(self.sequences_and_labels.keys())
        self.pfam_names = list(self.labels_and_sequences.keys())

    def __len__(self):
        return len(self.sequences_and_labels)

    def __getitem__(self, idx):
        x = self._sample_example()
        return x

class ProteinSequenceDataset(torch.utils.data.Dataset):

    def __init__(self,
            json_files,
            max_sequence_length,
            encode_as_image,
            multilabel,
            name_to_label_mapping,
            n_classes=None
            ):

        self.max_sequence_length = max_sequence_length
        self.multilabel = multilabel
        self.encode_as_image = encode_as_image
        self.name_to_label_mapping = name_to_label_mapping

        if n_classes is None:
            self.n_classes = get_n_classes(self.name_to_label_mapping)
        else:
            self.n_classes = n_classes

        self._build_dataset(json_files)

    def _encoding_func(self, x):
        # TODO: implement more logic here to use variable encodings.

        labels, seq = x

        oh = encode_protein_as_one_hot_vector(seq, self.max_sequence_length)
        if not self.encode_as_image:
            oh = np.argmax(oh, axis=-1)

        if self.multilabel:
            label = np.zeros((self.n_classes, ))
            label[np.asarray(labels)] = 1
            labels = label

        return [oh, label]


    def _build_dataset(self, json_files):
        if not isinstance(json_files, list):
            json_files = [json_files]

        if len(json_files) == 1:
            self.sequences_and_labels = read_sequences_from_json(json_files[0], self.name_to_label_mapping)
        else:
            self.sequences_and_labels = []
            for j in json_files:
                self.sequences_and_labels.extend(read_sequences_from_json(j,
                    self.name_to_label_mapping))

        shuffle(self.sequences_and_labels)


    def __len__(self):

        return len(self.sequences_and_labels)


    def __getitem__(self, idx):

        x, y = self._encoding_func(self.sequences_and_labels[idx])
        return torch.tensor(x.squeeze()).transpose(-1, -2).float(), torch.tensor(y) 


def get_n_classes(name_to_label_mapping):

    with open(name_to_label_mapping, 'r') as f:
        dct = json.load(f)
    s = set()
    for accession_id in dct.values():
        s.add(accession_id)

    n_classes = len(s)
    return n_classes



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dirs = ['profmark0.2','profmark0.3','profmark0.4','profmark0.5','profmark0.6','profmark0.7',
          'profmark0.8','profmark0.9']

    root ='../../data/pmark-outputs/profmark0.6/json/train-sequences-and-labels.json' 
    dset = Word2VecStyleDataset(root, 256,
            '../../data/pmark-outputs/profmark0.6/json/name-to-label.json')

    s = time.time()
    x, y,z = dset[1]
    dset = torch.utils.data.DataLoader(dset, batch_size=1024)
    for x, y, z in dset:
        print(x.shape, y.shape, z.shape)
