import json
import os
import pdb
import logging
from random import shuffle, seed

import numpy as np
import torch

from prefilter.utils.datasets import GSCC_SAVED_TF_MODEL_PATH

log = logging.getLogger(__name__)

seed(1)

__all__ = [
    "encode_protein_as_one_hot_vector",
    "parse_labels",
    "pad_batch",
    "stack_batch",
    "PROT_ALPHABET",
    "LEN_PROTEIN_ALPHABET",
    "fasta_from_file",
]

PROT_ALPHABET = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "X": 20,
    "Y": 21,
    "Z": 22,
}

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)


def encode_protein_as_one_hot_vector(protein, maxlen=None):
    # input: raw protein string of arbitrary length
    # output: np.array() of size (1, maxlen, length_protein_alphabet)
    # Each row in the array is a separate character, encoded as
    # a one-hot vector

    protein = protein.upper().replace("\n", "")

    if maxlen is not None:
        one_hot_encoding = np.zeros((LEN_PROTEIN_ALPHABET, maxlen))
        protein = protein[:maxlen]
    else:
        one_hot_encoding = np.zeros((LEN_PROTEIN_ALPHABET, len(protein)))

    for i, residue in enumerate(protein):
        try:
            one_hot_encoding[PROT_ALPHABET[residue], i] = 1
        except KeyError:
            one_hot_encoding[PROT_ALPHABET["X"], i] = 1  # X is "any amino acid"

    return one_hot_encoding

def parse_labels(labelstring):
    """
    Parses the Pfam accession IDs from a > line in a fasta file.
    Assumes that the fasta files have been generated with prefilter.utils.label_fasta.
    Each > line of the fasta file should look like this:
    >arbitrary name of sequence | PFAMID1 PFAMID2 PFAMID3 ... PFAMIDN
    <sequence>
    Each sequence can have one or many pfam accession IDs as labels.
    :param labelstring: line to parse labels from
    :type labelstring: str
    :return: List of Pfam accession IDs
    :rtype: Union[List[str], None]
    """
    delim = labelstring.find("|")

    if delim == -1:
        return None

    labels = labelstring[delim + 1 :].split(" ")
    labels = list(filter(len, labels))

    if not len(labels):
        return None

    return labels


def fasta_from_file(fasta_file):
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []

    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        sequence_strs.append("".join(buf))
        cur_seq_label = None
        buf = []

    with open(fasta_file, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:  # sequence line
                buf.append(line.strip())

    _flush_current_seq()

    return sequence_labels, sequence_strs


def _pad_sequences(sequences):
    mxlen = np.max([s.shape[-1] for s in sequences])
    padded_batch = np.zeros((len(sequences), LEN_PROTEIN_ALPHABET, mxlen))
    masks = []
    for i, s in enumerate(sequences):
        padded_batch[i, :, : s.shape[-1]] = s
        mask = np.ones((1, mxlen))
        mask[:, : s.shape[-1]] = 0
        masks.append(mask)

    masks = np.stack(masks)
    return torch.tensor(padded_batch).float(), torch.tensor(masks).bool()


def pad_batch(batch):
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    features, features_mask = _pad_sequences(features)
    return features, features_mask, torch.stack(labels)


def stack_batch(batch):
    """replicates default collate_fn for API consistency"""
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return torch.stack(features), torch.stack(labels)
