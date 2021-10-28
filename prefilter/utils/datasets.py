import os
import json
import time
import torch
import numpy as np
from collections import defaultdict

import prefilter.utils as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

__all__ = ['ProteinSequenceDataset',
           'SimpleSequenceIterator',
           'fasta_from_file',
           'data_run']

GSCC_SAVED_TF_MODEL_PATH = '/home/tc229954/data/prefilter/proteinfer/trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760'


class ProteinSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, fasta_files,
                 existing_name_to_label_mapping=None,
                 sample_sequences_based_on_family_membership=False,
                 sample_sequences_based_on_num_labels=False,
                 use_pretrained_model_embeddings=False,
                 evaluating=False):

        self.fasta_files = fasta_files

        if not len(self.fasta_files):
            raise ValueError("No fasta files found")

        self.existing_name_to_label_mapping = existing_name_to_label_mapping
        self.evaluating = evaluating
        self.sample_sequences_based_on_family_membership = sample_sequences_based_on_family_membership
        self.sample_sequences_based_on_num_labels = sample_sequences_based_on_num_labels
        self.use_pretrained_model_embeddings = use_pretrained_model_embeddings

        self._build_dataset()

    def _encoding_func(self, x):
        return utils.encode_protein_as_one_hot_vector(x.upper())

    def _build_dataset(self):

        self.sequences_and_labels = []

        if self.existing_name_to_label_mapping is None:
            self.name_to_class_code = {}
            class_id = 0

        elif isinstance(self.existing_name_to_label_mapping, str):
            s = 'loading class mapping from file {}'.format(self.existing_name_to_label_mapping)
            print(s)

            with open(self.existing_name_to_label_mapping, 'r') as src:
                self.name_to_class_code = json.load(src)

            class_id = len(self.name_to_class_code)

        elif isinstance(self.existing_name_to_label_mapping, dict):
            self.name_to_class_code = self.existing_name_to_label_mapping
            class_id = len(self.name_to_class_code)

        else:
            s = 'expected existing_name_to_label_mapping to be one of dict, string, or None, found {}'.format(
                type(self.existing_name_to_labelmapping))
            raise ValueError(s)

        self.family_to_indices = defaultdict(list)

        index_counter = 0

        for f in self.fasta_files:

            with open(f, 'r') as src:
                sequence_labels, sequences = fasta_from_file(f)

            sequence_to_labels = defaultdict(list)

            for sequence, labelstring in zip(sequences, sequence_labels):
                labels = labelstring[labelstring.find("|")+1:].split(" ")
                labels = list(filter(len, labels))
                sequence_to_labels[sequence] = labels

            for sequence, labelset in sequence_to_labels.items():

                for label in labelset:

                    self.family_to_indices[label].append(index_counter)

                    if label not in self.name_to_class_code:

                        if not self.evaluating:
                            self.name_to_class_code[label] = class_id
                            class_id += 1

                index_counter += 1
                self.sequences_and_labels.append([labelset, sequence])

        if self.sample_sequences_based_on_family_membership:

            total_seq = sum(map(len, self.family_to_indices.values()))
            family_to_frequency = {k: len(v) / total_seq for k, v in self.family_to_indices.items()}
            family_to_resampled_membership = {}

            for family, indices in self.family_to_indices.items():
                indices = np.asarray(indices)
                keep_prob = np.sqrt(1e-5 / family_to_frequency[family])
                kept = np.count_nonzero(np.random.rand(len(indices)) <= keep_prob)

                if kept == 0:
                    family_to_resampled_membership[family] = 1
                else:
                    family_to_resampled_membership[family] = kept

            self.length_of_dataset = sum(list(family_to_resampled_membership.values()))
            self.families = np.asarray(list(family_to_resampled_membership.keys()))
            self.sample_probs = np.asarray(list(family_to_resampled_membership.values()))
            self.sample_probs = self.sample_probs / np.sum(self.sample_probs)

        self.sequences_and_labels = np.asarray(self.sequences_and_labels, dtype='object')
        if self.sample_sequences_based_on_num_labels:

            self.family_to_sample_dist = {}
            for family in self.families:
                lengths_of_label_sets = list(map(lambda x: len(x[0]),
                                                 self.sequences_and_labels[self.family_to_indices[family]]))
                x = np.asarray(lengths_of_label_sets) ** 4
                self.family_to_sample_dist[family] = x / np.sum(x)

        self.n_classes = class_id + 1

    def __len__(self):
        if self.sample_sequences_based_on_family_membership:
            return self.length_of_dataset
        else:
            return len(self.sequences_and_labels)

    def _make_multi_hot(self, labels):
        y = np.zeros(self.n_classes)
        class_ids = [self.name_to_class_code[l] for l in labels]
        for idx in class_ids:
            y[idx] = 1
        return torch.as_tensor(y)

    def __getitem__(self, idx):

        if self.sample_sequences_based_on_family_membership:
            family = np.random.choice(self.families, p=self.sample_probs)
            family_indices = self.family_to_indices[family]
            if self.sample_sequences_based_on_num_labels:
                idx = np.random.choice(family_indices, p=self.family_to_sample_dist[family])
                labels, features = self.sequences_and_labels[idx]
            else:
                idx = int(np.random.rand() * len(family_indices))
                labels, features = self.sequences_and_labels[family_indices[idx]]
        else:
            labels, features = self.sequences_and_labels[idx]

        x = self._encoding_func(features)
        y = self._make_multi_hot(labels)
        return torch.as_tensor(x), y


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


class SimpleSequenceIterator(torch.utils.data.Dataset):

    def __init__(self, fasta_file, one_hot_encode=False):
        """
        takes a fasta file as input
        """

        self.fasta_file = fasta_file
        self.labels, self.sequences = fasta_from_file(fasta_file)
        self.one_hot_encode = one_hot_encode

    def _encoding_func(self, x):
        if self.one_hot_encode:
            return torch.as_tensor(utils.encode_protein_as_one_hot_vector(x.upper()))
        else:
            return x

    def __getitem__(self, idx):
        return self._encoding_func(self.sequences[idx]), torch.as_tensor([0])

    def __len__(self):
        return len(self.sequences)


def data_run():
    from glob import glob

    fasta_files = glob('/home/tc229954/tmp/*train*')

    dataset = ProteinSequenceDataset(fasta_files,
                                     sample_sequences_based_on_family_membership=True,
                                     sample_sequences_based_on_num_labels=True)
    i = 0
    for i in range(len(dataset)):
        print(len(np.where(dataset[i][1] == 1)[0]))


if __name__ == '__main__':
    data_run()
