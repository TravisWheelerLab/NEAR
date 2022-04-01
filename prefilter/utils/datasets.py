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

import yaml

import prefilter
import prefilter.utils as utils
from prefilter import DECOY_FLAG
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__all__ = [
    "SequenceDataset",
    "DecoyIterator",
    "SequenceIterator",
    "RankingIterator",
    "ContrastiveGenerator",
]


# this could be sped up if i did it vectorized
# but whatever for now
def _compute_soft_label(e_value: float):
    if e_value < 1e-30:
        e_value = 1e-30
    # take care of underflow / inf values in the np.log10
    x = np.clip(np.floor(np.log10(e_value)) * -1, 0, 5) / 5
    return x


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
        if isinstance(labels, str):
            labels = [labels]
        class_ids = []
        for label in labels:

            if isinstance(label, list) and len(label) != 1:
                label = label[0]

            if label == DECOY_FLAG:
                return []

            class_ids.append(self.name_to_class_code[label])

        return class_ids

    def _make_multi_hot(self, labels):
        y = np.zeros(self.n_classes)
        class_ids = self._class_id_vector(labels)
        for idx in class_ids:
            y[idx] = 1
        return torch.as_tensor(y)

    def _make_distillation_vector(self, labels, e_values):
        y = np.zeros(self.n_classes)
        class_ids = self._class_id_vector(labels)
        for idx, e_value in zip(class_ids, e_values):
            y[idx] = _compute_soft_label(float(e_value))
        return torch.as_tensor(y, dtype=torch.float32, device=torch.device("cpu"))

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


class SequenceIterator(SequenceDataset):
    """
    Iterates over the sequences in a/the fasta file(s) and encodes them
    for ingestion into an ml algorithm.
    """

    def __init__(
        self,
        fasta_files,
        name_to_class_code,
        evalue_threshold=None,
        distillation_labels=False,
    ):

        super().__init__(fasta_files, name_to_class_code)

        self.sequences_and_labels = []
        self.distillation_labels = distillation_labels
        self.evalue_threshold = evalue_threshold

        self._build_dataset()

    def _build_dataset(self) -> None:
        # TODO: Write rust extension for this.
        for fasta_file in self.fasta_files:
            print(os.path.basename(fasta_file))
            labels, sequences = utils.fasta_from_file(fasta_file)
            for labelstring, sequence in zip(labels, sequences):
                labelset = utils.parse_labels(labelstring)
                if self.evalue_threshold is not None:
                    new_labelset = []
                    for example in labelset:
                        if float(example[-1]) <= self.evalue_threshold:
                            new_labelset.append(example)
                    labelset = new_labelset
                if not len(labelset):
                    raise ValueError(
                        f"Line in {fasta_file} does not contain any labels. Please fix."
                    )

                if len(labelset) == 1:
                    if labelset[0] == "DECOY":
                        # no labels for decoys
                        lvec = torch.as_tensor(np.zeros(self.n_classes))
                    elif len(labelset[0]) != 4:
                        # it's an emission sequence
                        lvec = self._make_distillation_vector(labelset, [1e-30])
                    else:
                        # its eeal but only has one label
                        lvec = self._make_distillation_vector(
                            [l[0] for l in labelset], [l[-1] for l in labelset]
                        )
                else:
                    lvec = self._make_distillation_vector(
                        [l[0] for l in labelset], [l[-1] for l in labelset]
                    )

                self.sequences_and_labels.append([sequence, lvec])

    def __getitem__(self, idx):
        features, labels = self.sequences_and_labels[idx]
        encoded_features = self._encoding_func(features)
        return torch.as_tensor(encoded_features), labels

    def __len__(self):
        return len(self.sequences_and_labels)

    def shuffle(self):
        shuffle(self.sequences_and_labels)


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
    def __init__(
        self, fasta_files, name_to_class_code, max_labels_per_seq, evalue_threshold
    ):
        super().__init__(fasta_files, name_to_class_code)

        self._build_dataset(max_labels_per_seq, evalue_threshold)

    def _build_dataset(self, max_labels_per_seq, evalue_threshold):
        self.labels_and_sequences = utils.load_sequences_and_labels(
            self.fasta_files, max_labels_per_seq, evalue_threshold
        )

    def __getitem__(self, idx):
        labelset, sequence = self.labels_and_sequences[idx]
        return self._encoding_func(sequence), self._make_multi_hot(labelset), labelset

    def __len__(self):
        return len(self.labels_and_sequences)


class SequenceIterator(SequenceDataset):
    def __init__(
        self, fasta_files, name_to_class_code, max_labels_per_seq, evalue_threshold
    ):
        super().__init__(fasta_files, name_to_class_code)

        self._build_dataset(max_labels_per_seq, evalue_threshold)

    def _build_dataset(self, max_labels_per_seq, evalue_threshold):
        self.labels_and_sequences = utils.load_sequences_and_labels(
            self.fasta_files, max_labels_per_seq, evalue_threshold
        )

    def __getitem__(self, idx):
        labelset, sequence = self.labels_and_sequences[idx]
        return self._encoding_func(sequence), labelset

    def __len__(self):
        return len(self.labels_and_sequences)


class ContrastiveGenerator(SequenceDataset):
    def __init__(
        self,
        fasta_files,
        logo_path,
        name_to_class_code,
        oversample_neighborhood_labels,
        oversample_freq=5,
    ):

        if not len(fasta_files):
            raise ValueError("No fasta files found")

        self.fasta_files = fasta_files
        self.logo_path = logo_path
        self.name_to_sequences = defaultdict(list)
        self.name_to_class_code = name_to_class_code
        self.oversample_freq = oversample_freq
        self.name_to_logo = {}
        self.names = None
        self.oversample_neighborhood_labels = oversample_neighborhood_labels
        if self.oversample_neighborhood_labels:
            self.neighborhood_name_to_sequences = defaultdict(list)
            self.neighborhood_names = None
            self.step_count = 0

        if not isinstance(self.fasta_files, list):
            self.fasta_files = [self.fasta_files]

        self._build_dataset()

    def _build_dataset(self):

        with open(prefilter.name_to_accession_id, "r") as src:
            name_to_acc_id = yaml.safe_load(src)

        acc_id_to_name = {v: k for k, v in name_to_acc_id.items()}
        del name_to_acc_id

        for fasta_file in self.fasta_files:
            print(f"loading sequences from {fasta_file}")
            labels, sequences = utils.fasta_from_file(fasta_file)
            for labelstring, sequence in zip(labels, sequences):
                # parse labels
                labelset = utils.parse_labels(labelstring)
                labelset = list(filter(lambda x: float(x[-1]) < 1e-5, labelset))
                labelset = [lab[0] for lab in labelset]

                for k, label in enumerate(labelset):
                    if self.oversample_neighborhood_labels:
                        if k == 0:
                            self.name_to_sequences[label].append(
                                utils.encode_protein_as_one_hot_vector(sequence)
                            )
                        else:
                            self.neighborhood_name_to_sequences[label].append(
                                utils.encode_protein_as_one_hot_vector(sequence)
                            )
                    else:
                        self.name_to_sequences[label].append(
                            utils.encode_protein_as_one_hot_vector(sequence)
                        )
        # now construct the name-to-logo by
        # iterating over the pfam accession IDs in name to sequences
        if self.oversample_neighborhood_labels:
            accession_ids = set(
                list(self.name_to_sequences.keys())
                + list(self.neighborhood_name_to_sequences.keys())
            )
        else:
            accession_ids = list(self.name_to_sequences.keys())

        for accession_id in accession_ids:
            logo_name = acc_id_to_name[accession_id]
            logo_file = os.path.join(self.logo_path, f"{logo_name}.0.5-train.hmm.logo")
            if not os.path.isfile(logo_file):
                raise ValueError(f"No logo file found at {logo_file}")
            else:
                self.name_to_logo[accession_id] = utils.logo_from_file(logo_file)

        if self.oversample_neighborhood_labels:
            self.names = list(self.name_to_sequences.keys())
            self.neighborhood_names = list(self.neighborhood_name_to_sequences.keys())
        else:
            self.names = list(self.name_to_sequences.keys())

        self.len = sum(list(map(len, list(self.name_to_sequences.values()))))

    def __len__(self):
        return self.len

    def _sample(self, name_to_sequences, names):
        """
        Helper function to sample from a dict mapping pfam accession id to a list of sequences
        :param dct:
        :type dct:
        :return:
        :rtype:
        """
        name_idx = int(np.random.rand() * len(names))
        name = names[name_idx]
        pos_seqs, pos_logo = name_to_sequences[name], self.name_to_logo[name]
        seq_idx = int(np.random.rand() * len(pos_seqs))
        pos_seq = pos_seqs[seq_idx]
        return pos_seq, pos_logo, self.name_to_class_code[name]

    def __getitem__(self, idx):
        # stochastic
        if self.oversample_neighborhood_labels:
            if self.step_count % self.oversample_freq == 0:
                pos_seq, pos_logo, class_code = self._sample(
                    self.name_to_sequences, self.names
                )
            else:
                pos_seq, pos_logo, class_code = self._sample(
                    self.neighborhood_name_to_sequences, self.neighborhood_names
                )
            self.step_count += 1
        else:
            pos_seq, pos_logo, class_code = self._sample(
                self.name_to_sequences, self.names
            )

        return pos_seq, pos_logo, class_code


if __name__ == "__main__":
    from glob import glob

    fpath = "/home/tc229954/max_hmmsearch/200_file_subset/*fa"
    fs = glob(fpath)[:10]
    name_to_class_code = utils.create_class_code_mapping(fs)

    psd = ContrastiveGenerator(
        fasta_files=fs,
        logo_path="/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/",
        name_to_class_code=name_to_class_code,
        oversample_neighborhood_labels=True,
    )

    psd = torch.utils.data.DataLoader(
        psd,
        batch_size=33,
        collate_fn=utils.pad_contrastive_batches,
    )

    for x, y, z in psd:
        print(x.shape, y.shape, z.shape)
