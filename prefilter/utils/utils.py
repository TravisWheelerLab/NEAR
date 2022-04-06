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
    "load_sequences_and_labels",
    "encode_protein_as_one_hot_vector",
    "parse_labels",
    "pad_features_in_batch",
    "pad_labels_and_features_in_batch",
    "stack_batch",
    "PROT_ALPHABET",
    "inverse",
    "pad_batch_with_labels",
    "LEN_PROTEIN_ALPHABET",
    "handle_figure_path",
    "fasta_from_file",
    "create_class_code_mapping",
    "msa_from_file",
    "encode_msa",
    "pad_contrastive_batches",
    "logo_from_file",
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
}
inverse = {v: k for k, v in PROT_ALPHABET.items()}

LEN_PROTEIN_ALPHABET = len(PROT_ALPHABET)


def encode_msa(msa_seqs: List[List[str]]):
    # CxHxW
    out = np.zeros((len(PROT_ALPHABET), len(msa_seqs), len(msa_seqs[0])))
    for i, seq in enumerate(msa_seqs):
        single_encoding = encode_protein_as_one_hot_vector(seq)
        out[:, i, :] = single_encoding
    return out


def load_sequences_and_labels(
    fasta_files: List[str],
    max_labels_per_seq: int = None,
    evalue_threshold: float = None,
) -> List[Tuple[List[str], str]]:
    """
    :param evalue_threshold:
    :type evalue_threshold:
    :param max_labels_per_seq:
    :type max_labels_per_seq:
    :param fasta_files:
    :type fasta_files:
    :return: List of [labels, sequence].
    :rtype:
    """
    labels_to_sequence = []
    for fasta in fasta_files:
        labelset, sequences = fasta_from_file(fasta)
        # parse labels, get
        for labelstring, sequence in zip(labelset, sequences):
            labels = parse_labels(labelstring)
            if evalue_threshold is not None:
                labels = list(
                    filter(lambda x: float(x[-1]) <= evalue_threshold, labels)
                )
            if max_labels_per_seq is not None:
                labels = labels[:max_labels_per_seq]
            if labels is None:
                print(labelstring)
                continue
            else:
                labels_to_sequence.append([labels, sequence])

    return labels_to_sequence


def msa_from_file(msa_filepath: str):
    msa_ext = os.path.splitext(msa_filepath)[1]
    if msa_ext != ".afa":
        raise ValueError(f"Ext must be .afa, got {msa_ext}")

    labels, sequences = fasta_from_file(msa_filepath)
    return labels, sequences


def logo_from_file(logo_filepath: str):
    logo_ext = os.path.splitext(logo_filepath)[1]
    if logo_ext != ".logo":
        raise ValueError(f"Ext must be .logo, got {logo_ext}")
    logo = np.genfromtxt(logo_filepath, skip_header=2)[:, 1:-2]
    return logo.T


def handle_figure_path(figure_path: str, ext: str = ".png") -> str:
    bs = os.path.basename(figure_path)
    name, curr_ext = os.path.splitext(bs)

    if len(curr_ext) == 0:
        figure_path = figure_path + ext

    return figure_path


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
            # AA doesn't exist or is unknown
            pass
            # one_hot_encoding[PROT_ALPHABET["X"], i] = 1  # X is "any amino acid"

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


def create_class_code_mapping(fasta_files, evalue_threshold=1e-5):
    """
    TODO: CONVERT TO RUST.
    in order to create the mapping on the fly we have to load every sequence in our dataset!
    This can take a long time. Is there a faster way to do it?
    :param fasta_files:
    :type fasta_files:
    :return:
    :rtype:
    """

    name_to_class_code = {}

    class_code = 0
    for fasta_file in fasta_files:
        # print(f"loading {fasta_file}, {len(name_to_class_code)} unique names so far")
        labels, sequences = fasta_from_file(fasta_file)
        for label, sequence in zip(labels, sequences):

            labelset = parse_labels(label)

            if labelset is None or len(labelset) == 0:
                raise ValueError(
                    f"Line in {fasta_file} does not contain any labels. Please fix."
                )
            else:

                for name in labelset:
                    # if we have an accession id plus two coordinates;
                    # grab only the name
                    if len(name) == 4:
                        if float(name[-1]) > evalue_threshold:
                            break

                    if isinstance(name, list):
                        name = name[0]
                    # otherwise, don't mess with it.
                    if name not in name_to_class_code:
                        name_to_class_code[name] = class_code
                        class_code += 1

    return name_to_class_code


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


def _pad_sequences(sequences, minlen=60):
    mxlen = np.max([s.shape[-1] for s in sequences])
    if mxlen < minlen:
        mxlen = minlen
    padded_batch = np.zeros((len(sequences), LEN_PROTEIN_ALPHABET, mxlen))
    masks = []
    for i, s in enumerate(sequences):
        padded_batch[i, :, : s.shape[-1]] = s
        mask = np.ones((1, mxlen))
        mask[:, : s.shape[-1]] = 0
        masks.append(mask)

    masks = np.stack(masks)
    return torch.tensor(padded_batch).float(), torch.tensor(masks).bool()


def _pad_labels(labels):
    mxlen = np.max([l.shape[-1] for l in labels])
    padded_batch = np.ones((len(labels), labels[0].shape[0], mxlen)) * MASK_FLAG
    for i, s in enumerate(labels):
        padded_batch[i, :, : s.shape[-1]] = s
    return torch.tensor(padded_batch).float()


def pad_features_in_batch(batch):
    """
    Only pad the features in the batch.
    Return the padded features and the corresponding mask plus stacked labels.
    :param batch:
    :type batch:
    :return:
    :rtype:
    """
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    features, features_mask = _pad_sequences(features)
    return features, features_mask, labels


def pad_contrastive_batches(batch):
    """
    Pad batches that consist of a 3-tuple: seq, logo, and label
    :param batch: list of np.ndarrays encoding protein sequences/logos
    :type batch: List[np.ndarray]
    :return: torch.tensor
    :rtype: torch.tensor
    """

    seqs = [b[0] for b in batch]
    logos = [b[1] for b in batch]
    data = seqs + logos
    labels = [b[2] for b in batch]
    seqs, seqs_mask = _pad_sequences(data)

    return torch.as_tensor(seqs), torch.as_tensor(seqs_mask), torch.as_tensor(labels)


def pad_labels_and_features_in_batch(batch):
    """
    Pad both the labels and sequence to the max length in the batch.
    Useful for training FCNNs on sequence data.
    :param batch: Batch to pad.
    :type batch:
    :return:
    :rtype:
    """
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    features, features_mask = _pad_sequences(features)
    labels = _pad_labels(labels)
    return features, features_mask, labels


def pad_batch_with_labels(batch):
    """
    Pad the features in the batch to the max sequence length in the batch.
    Returns a mask for non-data features.
    Labels should be vectors of uniform size (not fcnn labels).
    Also returns the string labels.
    :param batch: Batch to pad.
    :type batch: List[Tuple[np.ndarray, np.ndarray, List[str]]
    :return: features, mask for features, labels, and string labels
    :rtype: Tuple[torch.tensor, torch.tensor, torch.tensor, List[str]]
    """
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    string_labels = [b[2] for b in batch]
    features, features_mask = _pad_sequences(features)
    return features, features_mask, torch.stack(labels), string_labels


def stack_batch(batch):
    features = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return torch.stack(features), torch.stack(labels)
