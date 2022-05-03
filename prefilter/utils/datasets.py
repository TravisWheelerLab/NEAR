# pylint: disable=no-member
import os
import matplotlib.pyplot as plt
import pdb
import json
import time
from abc import ABC

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

__all__ = ["RealisticAliPairGenerator", "SwissProtGenerator", "ClusterIterator"]


def _remove_gaps(seq, label):
    _seq = []
    _label = []
    for s, l in zip(seq, label):
        if s not in ("-", "."):
            _seq.append(s)
            _label.append(l)
    return _seq, _label


def _sanitize_sequence(sequence):
    """
    Remove bad characters from sequences.
    :param sequence:
    :type sequence:
    :return:
    :rtype:
    """
    sanitized = []
    for char in sequence:
        char = char.upper()
        if char not in ("X", "B", "U", "O", "Z"):
            sanitized.append(char)
        else:
            sampled_char = utils.amino_alphabet[
                utils.amino_distribution.sample().item()
            ]
            sanitized.append(sampled_char)

    return sanitized


class SwissProtGenerator:
    def __init__(self, fa_file, apply_indels, minlen=256, training=True):

        self.fa_file = fa_file
        self.apply_indels = apply_indels
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s[:minlen] for s in seqs if minlen < len(s)]
        self.training = training
        self.sub_dists = utils.generate_correct_substitution_distributions()
        self.sub_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.indel_probs = [0.01, 0.05, 0.1, 0.15]
        shuffle(self.seqs)

    def __len__(self):
        if self.training:
            return len(self.seqs)
        else:
            return 10000

    def shuffle(self):
        shuffle(self.seqs)

    def __getitem__(self, idx):
        if not self.training:
            if idx == 0:
                print("shuffling.")
                self.shuffle()

            idx = np.random.randint(0, len(self.seqs))

        s1 = _sanitize_sequence(self.seqs[idx])

        s1 = torch.tensor([utils.char_to_index[c] for c in s1])
        n_subs = int(
            len(s1) * self.sub_probs[np.random.randint(0, len(self.sub_probs))]
        )

        if self.apply_indels:
            n_indels = int(
                len(s1) * self.indel_probs[np.random.randint(0, len(self.indel_probs))]
            )
        else:
            n_indels = None

        s2 = utils.mutate_sequence_correct_probabilities(
            sequence=s1,
            indels=n_indels,
            substitutions=n_subs,
            sub_distributions=self.sub_dists,
            aa_dist=utils.amino_distribution,
        )

        return s1, s2, idx % len(self.seqs)


class ClusterIterator:
    """
    I'm going to add in alignments here so we can easily look
    at them without grepping or anything.
    """

    def __init__(
        self,
        afa_files,
        min_seq_len,
        representative_index,
        evaluate_on_clustered_split,
        n_seq_per_target_family,
        use_test=True,
    ):

        if n_seq_per_target_family is not None and not evaluate_on_clustered_split:
            raise ValueError(
                "Can't have more than a single rep. sequence if evaluating on non-clustered data."
            )

        self.train_afa_files = afa_files
        self.min_seq_len = min_seq_len
        self.representative_index = representative_index
        self.n_seq_per_target_family = n_seq_per_target_family
        self.use_test = use_test

        self.seed_sequences = []
        self.seed_labels = []
        self.seed_gapped_sequences = []

        self.query_sequences = []
        self.query_labels = []
        self.query_gapped_sequences = []
        # dictionary to map validation files to test files.
        self.valid_to_test = {}

        if evaluate_on_clustered_split:
            self.valid_files = []
            for file in self.train_afa_files:
                valid_file = file.replace("-train.fa", "-valid.fa")
                test_file = file.replace("-train.fa", "-test.fa")
                if os.path.isfile(valid_file):
                    self.valid_files.append(valid_file)
                if os.path.isfile(test_file):
                    self.valid_to_test[valid_file] = test_file

            self._build_clustered_dataset()

        else:
            self._build_dataset()

    def _build_clustered_dataset(self):

        label_index = 0

        for valid_file in self.valid_files:

            valid_headers, valid_seqs = utils.fasta_from_file(valid_file)
            valid_seqs = [
                s for s in valid_seqs if len(s.replace("-", "")) >= self.min_seq_len
            ]
            valid_ungapped_seqs = [s.replace("-", "") for s in valid_seqs]
            valid_ungapped_seqs = [s[: self.min_seq_len] for s in valid_ungapped_seqs]

            if valid_file in self.valid_to_test and self.use_test:
                test_file = self.valid_to_test[valid_file]
                test_headers, test_seqs = utils.fasta_from_file(test_file)
                test_seqs = [
                    s for s in test_seqs if len(s.replace("-", "")) >= self.min_seq_len
                ]
                test_ungapped_seqs = [s.replace("-", "") for s in test_seqs]
                test_ungapped_seqs = [s[: self.min_seq_len] for s in test_ungapped_seqs]

                valid_ungapped_seqs = valid_ungapped_seqs + test_ungapped_seqs
                valid_seqs = valid_seqs + test_seqs

            if len(valid_ungapped_seqs) > 1:
                # grab the train file
                # and process it to sequences that are > len(256).
                train_file = valid_file.replace("-valid.fa", "-train.fa")
                train_headers, train_seqs = utils.fasta_from_file(train_file)
                train_seqs = [
                    s for s in train_seqs if len(s.replace("-", "")) >= self.min_seq_len
                ]
                train_ungapped_seqs = [s.replace("-", "") for s in train_seqs]
                train_ungapped_seqs = [
                    s[: self.min_seq_len] for s in train_ungapped_seqs
                ]

                # grab a random sample from the train set.
                shuf_idx = np.random.choice(
                    np.arange(len(train_ungapped_seqs)),
                    size=len(train_ungapped_seqs),
                    replace=False,
                )

                train_ungapped_seqs = [train_ungapped_seqs[i] for i in shuf_idx]
                train_seqs = [train_seqs[i] for i in shuf_idx]

                if len(train_ungapped_seqs) > 1:
                    if self.n_seq_per_target_family is None:
                        self.seed_sequences.append(
                            train_ungapped_seqs[self.representative_index]
                        )
                        self.seed_gapped_sequences.append(
                            train_seqs[self.representative_index]
                        )
                        self.seed_labels.append(label_index)
                    else:

                        # print(f"Using {len(train_ungapped_seqs[:self.n_seq_per_target_family])} sequences in target
                        # DB for label {label_index} corresponding to {train_file}, {len(train_ungapped_seqs)}")

                        self.seed_sequences.extend(
                            train_ungapped_seqs[: self.n_seq_per_target_family]
                        )
                        self.seed_gapped_sequences.extend(
                            train_seqs[: self.n_seq_per_target_family]
                        )
                        self.seed_labels.extend(
                            [label_index]
                            * len(train_ungapped_seqs[: self.n_seq_per_target_family])
                        )

                    self.query_sequences.extend(valid_ungapped_seqs)
                    self.query_gapped_sequences.extend(valid_seqs)

                    self.query_labels.extend(
                        [label_index for _ in range(len(valid_ungapped_seqs))]
                    )

                    label_index += 1

        if len(self.seed_sequences) == 0 or len(self.query_sequences) == 0:
            print(
                f"No seed or query seqs over {self.min_seq_len}. Lengths of seed and query sequences: {len(self.seed_sequences)}, "
                f"{len(self.query_sequences)}"
            )
            exit()

        self.seed_sequences = [s[: self.min_seq_len] for s in self.seed_sequences]

    def _build_dataset(self):

        label_index = 0
        for fasta in self.train_afa_files:
            headers, seqs = utils.fasta_from_file(fasta)
            seqs = [s for s in seqs if len(s.replace("-", "")) >= self.min_seq_len]
            ungapped_seqs = [s.replace("-", "") for s in seqs]
            ungapped_seqs = [s[: self.min_seq_len] for s in ungapped_seqs]

            if len(ungapped_seqs) > 1:
                self.seed_sequences.append(ungapped_seqs[self.representative_index])
                self.seed_gapped_sequences.append(seqs[self.representative_index])
                self.seed_labels.append(label_index)

                self.query_sequences.extend(
                    ungapped_seqs[self.representative_index + 1 :]
                )
                self.query_gapped_sequences.extend(
                    seqs[self.representative_index + 1 :]
                )
                self.query_labels.extend(
                    [
                        label_index
                        for _ in range(
                            len(ungapped_seqs[self.representative_index + 1 :])
                        )
                    ]
                )

                label_index += 1

        self.seed_sequences = [s[: self.min_seq_len] for s in self.seed_sequences]

    def get_cluster_representatives(self):
        seeds = []
        for seed in self.seed_sequences:
            santitized = _sanitize_sequence(seed)
            seeds.append(
                torch.as_tensor([utils.char_to_index[s.upper()] for s in santitized])
            )

        return seeds, self.seed_labels

    def __len__(self):
        return len(self.query_sequences)

    def __getitem__(self, idx):

        qseq = self.query_sequences[idx][: self.min_seq_len]
        label = self.query_labels[idx]
        sanitized = _sanitize_sequence(qseq)
        seq = torch.as_tensor([utils.char_to_index[i.upper()] for i in sanitized])

        return seq, label, self.query_gapped_sequences[idx]


if __name__ == "__main__":
    from glob import glob

    pfam_files = glob(
        "/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/*-train.fa"
    )
    gen = ClusterIterator(pfam_files, 256, 0, evaluate_on_clustered_split=True)
