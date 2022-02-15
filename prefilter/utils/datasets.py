# pylint: disable=no-member
import os
import pdb
import json
import time
import torch
import numpy as np

from collections import defaultdict
from random import shuffle
from typing import List, Union, Tuple, Optional, Dict

import prefilter
import prefilter.utils as utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


__all__ = [
    "ProteinSequenceDataset",
    "SequenceDataset",
    "DecoyIterator",
    "SimpleSequenceIterator",
    "RankingIterator",
]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self, fasta_files: Union[str, List[str]], name_to_class_code: Dict[str, int]
    ) -> None:

        if not len(fasta_files):
            raise ValueError("No fasta files found")

        self.fasta_files = fasta_files
        self.name_to_class_code = name_to_class_code

        if not isinstance(self.fasta_files, list):
            self.fasta_files = [self.fasta_files]

    def _make_label_vector(self, labels, len_sequence):
        """
        Create a label vector with len_sequence separate label vectors.
        Each amino acid should have at least one label and up to N.
        :param labels: list of strings
        :type labels: List[str]
        :param len_sequence: length of sequence
        :type len_sequence: int
        :return:
        :rtype:
        """
        y = np.zeros((self.n_classes, len_sequence))
        for labelstring in labels:

            if isinstance(labelstring, list):
                if len(labelstring) == 3:
                    label, begin, end = labelstring
                elif len(labelstring) == 4:
                    label, begin, end, evalue = labelstring
            else:
                label = labelstring
                begin, end = 0, len_sequence

            y[self.name_to_class_code[label], int(float(begin)) : int(float(end))] = 1

        return torch.as_tensor(y)

    def _class_id_vector(self, labels):
        class_ids = []
        for label in labels:

            if isinstance(label, list):
                label = label[0]

            class_ids.append(self.name_to_class_code[label])

        return class_ids

    def _make_multi_hot(self, labels):
        y = np.zeros(self.n_classes)
        class_ids = self._class_id_vector(labels)
        for idx in class_ids:
            y[idx] = 1
        return torch.as_tensor(y)

    def _encoding_func(self, x):
        return utils.encode_protein_as_one_hot_vector(x.upper())

    def _build_dataset(self):
        raise NotImplementedError("Must build the dataset for each custom iterator.")

    def __len__(self):
        raise NotImplementedError("Must specify length for each custom iterator.")

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Must implement __getitem__ for each custom iterator."
        )

    @property
    def n_classes(self):
        return len(self.name_to_class_code)


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

    def __init__(self, n_seq_per_fam: Optional[int] = None, no_resample=True) -> None:
        """
        :param n_seq_per_fam: number of sequences to sample per family
        :type n_seq_per_fam: int
        """
        self.label_to_sequence = defaultdict(list)
        self.label_to_count = defaultdict(int)
        self.label_to_index = defaultdict(int)
        self.n_seq_per_fam = n_seq_per_fam
        self.no_resample = no_resample
        self.sequences_and_labels = []
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
        if isinstance(key, list):
            key = key[0]

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
        return self.sequences_and_labels[idx][1], self.sequences_and_labels[idx][0]

    def __len__(self):
        # length is the number of unique pfam accession ids in the fasta files
        # which is NOT the number of classes we actually classify in our final classification
        # layer
        return len(self.sequences_and_labels)

    def compute(self):
        for label, sequenceset in self.label_to_sequence.items():
            # sequenceset contains [[lab1, lab2..], sequence]
            self.sequences_and_labels.extend(sequenceset)
        shuffle(self.sequences_and_labels)

    def shuffle(self):
        """
        Use this at the end of an epoch to shuffle the ordering of names that you sample.
        :return: None
        :rtype: None
        """
        shuffle(self.sequences_and_labels)


class ProteinSequenceDataset(SequenceDataset):
    def __init__(
        self,
        fasta_files: str,
        name_to_class_code: Dict[str, int],
        n_seq_per_fam: Optional[int] = None,
        no_resample: bool = True,
        single_embedding: bool = True,
        n_emission_sequences: int = 50,
    ) -> None:

        super().__init__(fasta_files, name_to_class_code)

        self.label_to_sequence = LabelMapping(
            n_seq_per_fam=n_seq_per_fam, no_resample=no_resample
        )
        self.single_embedding = single_embedding
        self.n_emission_sequences = n_emission_sequences
        self._build_dataset()

    def _build_dataset(self):
        # going to choose labels from a dictionary.
        # LabelMapping is used to sample at training time.
        for fasta_file in self.fasta_files:
            labels, sequences = utils.fasta_from_file(fasta_file)

            if "emission" in os.path.basename(fasta_file):
                # cut off the emission sequences at n_emission_seqs
                labels = labels[: self.n_emission_sequences]
                sequences = sequences[: self.n_emission_sequences]

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

    def __getitem__(self, idx):
        labels, features = self.label_to_sequence.sample(idx)
        encoded_features = self._encoding_func(features)
        if self.single_embedding:
            y = self._make_multi_hot(labels)
        else:
            y = self._make_label_vector(labels, len(features))
        return torch.as_tensor(encoded_features), y


class SimpleSequenceIterator(SequenceDataset):
    """
    Iterates over the sequences in a/the fasta file(s) and encodes them
    for ingestion into an ml algorithm.
    """

    def __init__(self, fasta_files, name_to_class_code, single_embedding):

        super().__init__(fasta_files, name_to_class_code)

        self.sequences_and_labels = []
        self.single_embedding = single_embedding

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

    def __getitem__(self, idx):
        example = self.sequences_and_labels[idx]
        if self.single_embedding:
            return self._encoding_func(example[0]), self._make_multi_hot(example[1])
        else:
            return self._encoding_func(example[0]), self._make_label_vector(
                example[1], len(example[0])
            )

    def __len__(self):
        return len(self.sequences_and_labels)


class DecoyIterator(SequenceDataset):
    """
    Iterates over label-less fasta files (decoys, usually).
    """

    def __init__(self, fasta_files, name_to_class_code):
        super().__init__(fasta_files, name_to_class_code)

        self.sequences = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        for fasta_file in self.fasta_files:
            _, sequences = utils.fasta_from_file(fasta_file)
            self.sequences.extend(sequences)

    def __getitem__(self, idx):
        example = self.sequences[idx]
        return self._encoding_func(example[0]), torch.zeros(self.n_classes)

    def __len__(self):
        return len(self.sequences)


class RankingIterator(SequenceDataset):
    def __init__(self, fasta_files, name_to_class_code):
        super().__init__(fasta_files, name_to_class_code)

        self._build_dataset()

    def _build_dataset(self):
        self.labels_and_sequences = utils.load_sequences_and_labels(self.fasta_files)

    def __getitem__(self, idx):
        labelset, sequence = self.labels_and_sequences[idx]
        return self._encoding_func(sequence), self._make_multi_hot(labelset), labelset

    def __len__(self):
        return len(self.labels_and_sequences)


if __name__ == "__main__":
    from glob import glob

    fs = glob("/home/tc229954/domtblouts/*fa")[:10]
    name_to_class_code = utils.create_class_code_mapping(fs)
    print(len(name_to_class_code))
    psd = ProteinSequenceDataset(
        fasta_files=fs, name_to_class_code=name_to_class_code, single_embedding=False
    )
    psd = torch.utils.data.DataLoader(
        psd, batch_size=2, collate_fn=utils.pad_labels_and_features_in_batch
    )
    for x, y, z in psd:
        print(x.shape, y.shape, z.shape)
