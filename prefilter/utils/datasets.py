# pylint: disable=no-member
import os
import pdb
import json
import time

import prefilter
import torch
import numpy as np
import logging
from collections import defaultdict
from random import shuffle
from typing import List, Union, Tuple, Optional, Dict

import prefilter
import prefilter.utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

log = logging.getLogger(__name__)


class LabelMapping:

    """
    Container class that handles sampling from a set of labeled sequences.
    Pfam families have disparate number of sequences. This class samples from each family uniformly, in effect
    balancing the number of sequences per family.

    It accomplishes this by creating a data structure that maps pfam accession ID to sequence:
    mapping = {PF1111: [[seq1, labelset1], [seq2, labelset2]....[seqN, labelsetN], PF1112: [[seq1, labelset1], [seq2, labelset2]....[seqN, labelsetN],..}
    At "sample" time, it iterates over the pfam accession IDs and grabs the ith [sequence, labelset].
    A separate data structure is used to keep track of the index for each family.
    """

    def __init__(self, n_seq_per_fam: Optional[int] = None) -> None:
        """
        :param n_seq_per_fam: number of sequences to sample per family
        :type n_seq_per_fam: int
        """
        self.label_to_sequence = defaultdict(list)
        self.label_to_count = defaultdict(int)
        self.label_to_index = defaultdict(int)
        self.n_seq_per_fam = n_seq_per_fam
        self.names = None

    def __setitem__(self, key: str, value: List[Tuple[str, List[str]]]) -> None:
        """
        :param key: Pfam accession id.
        :type key: str
        :param value: List with the sequence as the first element and the set of labels assoc. with the sequence
        as the second element.
        :type value: List[str, List[str]].
        :return: None.
        :rtype: None.
        """
        self.label_to_sequence[key].append(value)
        self.label_to_count[key] += 1

    def __getitem__(self, key):
        return self.label_to_sequence[key]

    def sample(self, idx: int) -> Tuple[List[str], str]:
        """
        Return a labelset, sequence pair. If n_seq_per_fam is specified, only grab from the first
        n_seq_per_fam members in each family (useful for scaling the problem size down).

        :param idx: index of family to grab.
        :type idx: int
        :return: set of labels and sequence associated with that set.
        :rtype: Tuple[List[str], str]
        """
        name = self.names[idx % len(self.names)]
        if self.n_seq_per_fam is None:
            sequence, labelset = self.label_to_sequence[name][
                self.label_to_index[name] % len(self.label_to_sequence[name])
            ]
        else:
            divisor = (
                len(self.label_to_sequence[name])
                if len(self.label_to_sequence[name]) < self.n_seq_per_fam
                else self.n_seq_per_fam
            )
            sequence, labelset = self.label_to_sequence[name][
                self.label_to_index[name] % divisor
            ]
        self.label_to_index[name] += 1
        return labelset, sequence

    def __len__(self):
        # length is the number of unique pfam accession ids in the fasta files
        # which is NOT the number of classes we actually classify in our final classification
        # layer
        return len(self.names)

    def compute(self):
        summation = sum(list(self.label_to_count.values()))
        self.label_to_count = {k: v / summation for k, v in self.label_to_count.items()}
        self.names = list(self.label_to_sequence.keys())
        shuffle(self.names)
        # shuffle the ordering of sequences in each family
        x = {}
        for k, v in self.label_to_sequence.items():
            shuffle(v)
            x[k] = v

        self.label_to_sequence = x

    def shuffle(self):
        """
        Use this at the end of an epoch to shuffle the ordering of names that you sample.
        :return: None
        :rtype: None
        """
        shuffle(self.names)


class ProteinSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fasta_files: str,
        name_to_class_code: Optional[Dict[str, int]],
        n_seq_per_fam: Optional[int] = None,
    ) -> None:

        if not len(fasta_files):
            raise ValueError("No fasta files found")

        self.fasta_files = fasta_files

        if not isinstance(self.fasta_files, list):
            self.fasta_files = [self.fasta_files]

        if name_to_class_code is None:
            with open(prefilter.id_to_class_code, "r") as src:
                self.name_to_class_code = json.load(src)
        else:
            self.name_to_class_code = name_to_class_code

        self.label_to_sequence = LabelMapping(n_seq_per_fam=n_seq_per_fam)
        self._build_dataset()

    def _encoding_func(self, x):
        return utils.encode_protein_as_one_hot_vector(x.upper())

    def _build_dataset(self):
        # going to choose labels from a dictionary.
        # LabelMapping is used to sample from at training time.
        for fasta_file in self.fasta_files:
            labels, sequences = utils.fasta_from_file(fasta_file)
            for labelstring, sequence in zip(labels, sequences):
                labelset = utils.parse_labels(labelstring)
                if not len(labelset):
                    raise ValueError(
                        f"Line in {fasta_file} does not contain any labels. Please fix."
                    )
                else:
                    for label in labelset:
                        self.label_to_sequence[label] = [sequence, labelset]

        self.label_to_sequence.compute()

    def __len__(self):
        return len(self.label_to_sequence)

    def _make_multi_hot(self, labels):
        y = np.zeros(self.n_classes)
        class_ids = [self.name_to_class_code[l] for l in labels]
        for idx in class_ids:
            y[idx] = 1
        return torch.as_tensor(y)

    @property
    def n_classes(self):
        return len(self.name_to_class_code)

    def __getitem__(self, idx):
        labels, features = self.label_to_sequence.sample(idx)
        x = self._encoding_func(features)
        y = self._make_multi_hot(labels)
        return torch.as_tensor(x), y


class SimpleSequenceIterator(torch.utils.data.Dataset):

    """
    Iterates over the sequences in a/the fasta file(s) and encodes them
    for ingestion into an ml algorithm.
    """

    def __init__(
        self, fasta_files: str, name_to_class_code: Optional[Dict[str, int]] = None
    ) -> None:

        self.fasta_files = fasta_files
        self.sequences_and_labels = []

        if not isinstance(self.fasta_files, list):
            self.fasta_files = [self.fasta_files]

        if name_to_class_code is None:
            with open(prefilter.id_to_class_code, "r") as src:
                self.name_to_class_code = json.load(src)
        else:
            self.name_to_class_code = name_to_class_code

        self._build_dataset()

    def _build_dataset(self) -> None:
        # Could refactor to not keep seqs in mem (just load on the fly).
        for fasta_file in self.fasta_files:
            labels, sequences = utils.fasta_from_file(fasta_file)
            for labelstring, sequence in zip(labels, sequences):
                labelset = utils.parse_labels(labelstring)
                if not len(labelset):
                    raise ValueError(
                        f"Line in {fasta_file} does not contain any labels. Please fix."
                    )
                else:
                    self.sequences_and_labels.append([sequence, labelset])

    def _make_multi_hot(self, labels):
        y = np.zeros(self.n_classes)
        class_ids = [self.name_to_class_code[l] for l in labels]
        for idx in class_ids:
            y[idx] = 1
        return torch.as_tensor(y)

    @property
    def n_classes(self):
        return len(self.name_to_class_code)

    def _encoding_func(self, x):
        return torch.as_tensor(utils.encode_protein_as_one_hot_vector(x.upper()))

    def __getitem__(self, idx):
        example = self.sequences_and_labels[idx]
        return self._encoding_func(example[0]), self._make_multi_hot(example[1])

    def __len__(self):
        return len(self.sequences_and_labels)


if __name__ == "__main__":

    from glob import glob

    psd = ProteinSequenceDataset(
        fasta_files=glob("/home/tc229954/subset/training_data0.5/*train.fa")
    )
    print(len(psd))
    exit()
    for features, labels in psd:
        print(features.shape, labels.shape)
