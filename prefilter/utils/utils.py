import json
import os
import pdb
from random import shuffle, seed

import numpy as np
import torch

from prefilter.utils.datasets import GSCC_SAVED_TF_MODEL_PATH

seed(1)

__all__ = ['encode_protein_as_one_hot_vector',
           'pad_batch',
           'stack_batch',
           'PROT_ALPHABET',
           'LEN_PROTEIN_ALPHABET']

PROT_ALPHABET = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
                 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                 'T': 17, 'V': 18, 'W': 19, 'X': 20, 'Y': 21, 'Z': 22}

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)


def _read_fasta(fasta, label, return_index=True):

    def _parse_line(line):
        idx = line.find('\n')
        family = line[:idx]
        sequence = line[idx + 1:].replace('\n', '')
        if return_index:
            return [label, sequence]
        else:
            return sequence

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

    if maxlen is not None:
        one_hot_encoding = np.zeros((LEN_PROTEIN_ALPHABET, maxlen))
        protein = protein[:maxlen]
    else:
        one_hot_encoding = np.zeros((LEN_PROTEIN_ALPHABET, len(protein)))

    for i, residue in enumerate(protein):
        try:
            one_hot_encoding[PROT_ALPHABET[residue], i] = 1
        except KeyError:
            one_hot_encoding[PROT_ALPHABET['X'], i] = 1  # X is "any amino acid"

    return one_hot_encoding


def read_sequences_from_json(json_file, fout=None):
    """
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
    """

    with open(json_file, 'r') as f:
        sequence_to_label = json.load(f)

    name_to_integer_label = {}
    integer_label = 0

    if fout is None:
        fout = os.path.join(os.path.dirname(json_file), 'name-to-label.json')

    if os.path.isfile(fout):
        with open(fout, 'r') as f:
            name_to_integer_label = json.load(f)
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


def read_sequences_from_fasta(files, save_name_to_label=False, return_index=True):
    """
    returns list of all sequences in the fasta files.
    list is a list of two-element lists with the first element the class
    label and the second the protein sequence.

    Assumes that each fasta file has the class name of the proteins as its
    filename. TODO: implement logic that parses the fasta header to get a class
    name.

    does not take care of train/dev/val splits. This is trivially
    implemented.

    """
    name_to_label = {}
    cnt = 0
    sequences = []

    if not isinstance(files, list):
        files = [files]

    for f in files:
        seq = _read_fasta(f, cnt, return_index=return_index)
        name_to_label[os.path.basename(f)] = cnt
        cnt += 1
        if return_index:
            sequences.extend(filter(lambda x: len(x[1]), seq))
        else:
            sequences.extend(filter(lambda x: len(x), seq))

    if save_name_to_label:
        with open('name_to_class_code.json', 'w') as f:
            json.dump(name_to_label, f)

    shuffle(sequences)
    return sequences


def _pad_sequences(sequences):
    mxlen = np.max([s.shape[-1] for s in sequences])
    padded_batch = np.zeros((len(sequences), LEN_PROTEIN_ALPHABET, mxlen))
    masks = []
    for i, s in enumerate(sequences):
        padded_batch[i, :, :s.shape[-1]] = s
        mask = np.ones((1, mxlen))
        mask[:, :s.shape[-1]] = 0
        masks.append(mask)

    masks = np.stack(masks)
    return torch.tensor(padded_batch).float(), torch.tensor(masks).bool()


def pad_batch(batch):
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    features, features_mask = _pad_sequences(features)
    return features, features_mask, torch.stack(labels)


def stack_batch(batch):
    """ replicates default collate_fn for API consistency """
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return torch.stack(features), torch.stack(labels)


if __name__ == '__main__':
    pass
