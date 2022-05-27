# pylint: disable=no-member
import os
from Bio import SeqIO
import esm as esm
import matplotlib.pyplot as plt
import string
import pdb
import json
import time
from abc import ABC
from copy import deepcopy

import torch
import numpy as np

from collections import defaultdict
from random import shuffle
from typing import List, Union, Tuple, Optional, Dict
from glob import glob
from sys import stdout

import yaml

import prefilter
import prefilter.utils as utils
import prefilter.models as models
from prefilter import DECOY_FLAG
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__all__ = [
    "MSAGenerator",
    "SwissProtGenerator",
    "ClusterIterator"
]

class MSAGenerator:

    def __init__(self, afa_files):

        self.msa_to_seqs = defaultdict(list)
        for f in afa_files:
            bs = os.path.basename(f)
            for header, seq in esm.data.read_fasta(f):
                # replace the dang gap character.
                seq = seq.replace('.', "-")[:128]
                if len(seq):
                    self.msa_to_seqs[bs].append((header, seq.upper()))

        self.names = list(self.msa_to_seqs.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx % len(self.names)]
        return self.msa_to_seqs[name][:10], 0, self.msa_to_seqs[name][0], 0, idx

def _sanitize_sequence(sequence):
    """
    Remove bad/unknown/ambiguous characters from sequences.
    :param sequence:
    :type sequence:
    :return:
    :rtype:
    """
    sanitized = []
    for char in sequence:
        char = char.upper()
        if char in ("X", "U", "O"):
            sampled_char = utils.amino_alphabet[
                utils.amino_distribution.sample().item()
            ]
            sanitized.append(sampled_char)
        elif char == "B":
            if int(2 * np.random.rand()) == 1:
                sanitized.append("D")
            else:
                sanitized.append("N")
        elif char == "Z":
            if int(2 * np.random.rand()) == 1:
                sanitized.append("E")
            else:
                sanitized.append("Q")
        else:
            sanitized.append(char)

    return sanitized


class SwissProtGenerator:
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):

        self.fa_file = fa_file
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s[:minlen] for s in seqs if len(s) >= minlen]
        self.training = training
        self.sub_dists = utils.generate_correct_substitution_distributions()
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        shuffle(self.seqs)

    def __len__(self):
        if self.training:
            return len(self.seqs)
        else:
            return 10000

    def shuffle(self):
        shuffle(self.seqs)

    def _sample(self, idx):
        if not self.training:
            if idx == 0:
                print("shuffling.")
                self.shuffle()

            idx = np.random.randint(0, len(self.seqs))

        s1 = _sanitize_sequence(self.seqs[idx])

        s1 = torch.as_tensor([utils.char_to_index[c] for c in s1])
        n_subs = int(
            len(s1) * self.sub_probs[np.random.randint(0, len(self.sub_probs))]
        )

        s2 = utils.mutate_sequence_correct_probabilities(
            sequence=s1,
            indels=None,
            substitutions=n_subs,
            sub_distributions=self.sub_dists,
            aa_dist=utils.amino_distribution,
        )

        return s1, s2, idx % len(self.seqs)

    def __getitem__(self, idx):
        s1, s2, label = self._sample(idx)
        return s1, s2, label


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
        include_all_families,
        n_seq_per_target_family,
        transformer,
        return_alignments=False,
    ):

        self.train_afa_files = afa_files
        self.min_seq_len = min_seq_len
        self.representative_index = representative_index
        self.n_seq_per_target_family = n_seq_per_target_family
        self.transformer = transformer
        self.include_all_families = include_all_families
        self.return_alignments = return_alignments

        self.valid_afa_path = "/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_set_subsampled_by_what_hmmer_gets_right/afa"
        self.valid_fa_path = "/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_set_subsampled_by_what_hmmer_gets_right/fasta"

        self.seed_sequences = []
        self.seed_labels = []

        self.query_sequences = []
        self.query_labels = []
        # dictionary to map train files to validation files.
        self.train_to_valid = {}
        self.train_to_valid_afa = {}

        train_files_to_remove = []

        for file in self.train_afa_files:
            # get corresponding validation file;
            valid_file = os.path.basename(file)
            if "-1" in os.path.basename(file):
                valid_file = valid_file.replace("-1", "-2") + ".fa"
            else:
                valid_file = valid_file.replace("-2", "-1") + ".fa"

            valid_afa = os.path.join(self.valid_afa_path, valid_file).replace(".fa", "")
            valid_fa = os.path.join(self.valid_fa_path, valid_file)
            if os.path.isfile(valid_fa):
                self.train_to_valid[file] = valid_fa
                self.train_to_valid_afa[file] = valid_afa
            else:
                # print(f"Couldn't find valid file for {file}")
                if not include_all_families:
                    train_files_to_remove.append(file)

        for file in train_files_to_remove:
            self.train_afa_files.remove(file)

        self._build_clustered_dataset()

    def _build_clustered_dataset(self):

        label_index = 0
        self.seed_alignments = []
        self.query_alignments = []
        self.label_to_seed_alignment = {}

        for train_file in self.train_afa_files:
            # first, grab representative sequences
            _, train_seqs = utils.fasta_from_file(train_file)
            train_alignments = deepcopy(train_seqs)
            # no gaps here, forget about headers.
            _train_seqs = []
            _train_alignments = []

            for train_seq, train_ali in zip(train_seqs, train_alignments):
                if len(train_seq.replace(".", "")) > self.min_seq_len:
                    _train_seqs.append(train_seq.replace(".", "")[: self.min_seq_len])
                    _train_alignments.append(train_ali[: self.min_seq_len])

            train_seqs = _train_seqs
            train_alignments = _train_alignments

            if not (len(train_seqs)):
                continue

            # shuffle train seqs...?
            self.seed_sequences.extend(train_seqs[: self.n_seq_per_target_family])
            self.seed_alignments.extend(
                train_alignments[: self.n_seq_per_target_family]
            )
            self.seed_labels.extend(
                [label_index] * len(train_seqs[: self.n_seq_per_target_family])
            )
            self.label_to_seed_alignment[label_index] = train_alignments[
                : self.n_seq_per_target_family
            ]

            if train_file in self.train_to_valid:
                # add them into the validation set;
                valid_file = self.train_to_valid[train_file]
                valid_alignment_file = self.train_to_valid_afa[train_file]
                _, valid_seqs = utils.fasta_from_file(valid_file)
                _, valid_alignments = utils.fasta_from_file(valid_alignment_file)

                _valid_seqs = []
                _valid_alignments = []

                for valid_seq, valid_ali in zip(valid_seqs, valid_alignments):
                    if len(valid_seq.replace(".", "")) > self.min_seq_len:
                        _valid_seqs.append(
                            valid_seq.replace(".", "")[: self.min_seq_len]
                        )
                        _valid_alignments.append(valid_ali[: self.min_seq_len])

                valid_seqs = _valid_seqs
                valid_alignments = _valid_alignments

                if not (len(valid_seqs)):
                    # print(
                    #     f"No sequence > {self.min_seq_len} in {valid_file}. Skipping."
                    # )
                    pass
                else:
                    if len(valid_seqs) != len(valid_alignments):
                        pdb.set_trace()

                    self.query_sequences.extend(valid_seqs)
                    self.query_alignments.extend(valid_alignments)
                    self.query_labels.extend([label_index] * len(valid_seqs))

            label_index += 1

        if len(self.seed_sequences) == 0 or len(self.query_sequences) == 0:
            print(
                f"No seed or query seqs over {self.min_seq_len}. Lengths of seed and query sequences: {len(self.seed_sequences)}, "
                f"{len(self.query_sequences)}"
            )
            exit()

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

        return seq, label, self.query_sequences[idx]
