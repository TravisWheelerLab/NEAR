# pylint: disable=no-member
import json
import re
import os
import pdb
import logging
from random import shuffle, seed
from typing import Union, List, Tuple

import numpy as np
import torch
from prefilter import MASK_FLAG, DECOY_FLAG

log = logging.getLogger(__name__)

seed(1)

__all__ = [
    "encode_protein_as_one_hot_vector",
    "parse_labels",
    "PROT_ALPHABET",
    "INVERSE_PROT_MAPPING",
    "LEN_PROTEIN_ALPHABET",
    "handle_figure_path",
    "fasta_from_file",
    "pad_contrastive_batches_with_labelvecs",
    "mask_mask",
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
    "Y": 20,
}
INVERSE_PROT_MAPPING = {v: k for k, v in PROT_ALPHABET.items()}

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)


def handle_figure_path(figure_path: str, ext: str = ".png") -> str:
    bs = os.path.basename(figure_path)
    name, curr_ext = os.path.splitext(bs)

    if len(curr_ext) == 0:
        figure_path = figure_path + ext

    return figure_path


def encode_protein_as_one_hot_vector(protein):
    # input: raw protein string of arbitrary length
    # output: np.array() of size (1, maxlen, length_protein_alphabet)
    # Each row in the array is a separate character, encoded as
    # a one-hot vector

    protein = protein.upper().replace("\n", "")
    protein = protein.replace("-", "")

    one_hot_encoding = np.zeros((LEN_PROTEIN_ALPHABET, len(protein)))

    for i, residue in enumerate(protein):
        one_hot_encoding[PROT_ALPHABET[residue], i] = 1

    return one_hot_encoding


def parse_labels(labelstring: str) -> Union[List[str], None]:
    """
    Parses the Pfam accession IDs from a > line in a fasta file.
    Assumes that the fasta files have been generated with prefilter.utils.label_fasta.
    Each > line of the fasta file should look like this:
    >arbitrary name of sequence | PFAMID1 PFAMID2 PFAMID3 ... PFAMIDN
    <sequence>
    or
    >arbitrary name of sequence | PFAMID1 (begin1, end1) PFAMID2 (begin2, end2) PFAMID3 ... PFAMIDN
    <sequence>
    Each sequence can have one or many pfam accession IDs as labels.
    If the fasta header doesn't have a | or it has a | followed by nothing list,
    :param labelstring: line to parse labels from
    :type labelstring: str
    :return: List of Pfam accession IDs
    :rtype: Union[List[str], None]
    """
    begin_char = labelstring.find("|")

    if begin_char == -1:
        return [DECOY_FLAG]

    if "(" in labelstring:
        # labelstring: ACC_ID (BEGIN END E_VALUE)
        labels = (
            labelstring[begin_char + 1 :].replace(")", "").replace("(", "").split(" ")
        )
        labels = list(filter(len, labels))
        labelset = []

        for i in range(0, len(labels), 4):
            accession_id, begin, end, e_value = (
                labels[i],
                labels[i + 1],
                labels[i + 2],
                labels[i + 3],
            )
            labelset.append([accession_id, begin, end, e_value])
        labels = labelset
    else:
        labels = labelstring[begin_char + 1 :].split(" ")

    labels = list(filter(len, labels))

    if not len(labels):
        return None

    return labels


def afa_from_file(afa_file: str):
    """
    Parse a .afa file.
    :param afa_file:
    :type afa_file:
    :return:
    :rtype:
    """
    labels, seqs = fasta_from_file(afa_file)
    return seqs


def fasta_from_file(fasta_file: str) -> Union[None, List[Tuple[str, str]]]:
    """
    Returns labels and sequences.
    param fasta_file: fasta file to load sequences + labels from.
    :type fasta_file: str
    :return: Labels, sequences, or none.
    :rtype: Union[None, List[List[str], List[str]]]
    """
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


def mask_mask(mask):
    idxs = torch.sum(~mask, axis=-1).squeeze().detach()
    for i, idx in enumerate(idxs):
        mask[i, :, (idx - 1) :] = True
    return mask


def pad_contrastive_batches_with_labelvecs(batch):
    """
    Pad batches that consist of a 3-tuple: seq, logo, and label
    :param batch: list of np.ndarrays encoding protein sequences/logos
    :type batch: List[np.ndarray]
    :return: torch.tensor
    :rtype: torch.tensor
    """

    seqs = [b[0] for b in batch]
    logos = [b[1] for b in batch]
    lvec1 = [b[2] for b in batch]
    lvec2 = [b[3] for b in batch]
    data = seqs + logos
    labelvecs = lvec1 + lvec2
    labels = [b[4] for b in batch]
    return (
        torch.stack(data),
        None,
        labelvecs,
        torch.as_tensor(labels),
    )
