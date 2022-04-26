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


class SwissProtGenerator:
    def __init__(self, fa_file, minlen=256, training=True):

        self.fa_file = fa_file
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s[:minlen] for s in seqs if minlen < len(s)]
        self.training = training
        self.sub_dists = utils.generate_sub_distributions()
        shuffle(self.seqs)

    def __len__(self):
        if self.training:
            return len(self.seqs)
        else:
            return 1000

    def shuffle(self):
        shuffle(self.seqs)

    def __getitem__(self, idx):
        if not self.training:
            idx = np.random.randint(0, len(self.seqs))

        s1 = self.seqs[idx]
        _s1 = []
        for x in s1:
            x = x.upper()
            if x not in ("X", "B", "U", "O", "Z"):
                _s1.append(x)
            else:
                _s1.append(
                    utils.amino_alphabet[
                        np.random.randint(0, len(utils.amino_alphabet))
                    ]
                )

        s1 = _s1

        lvec1 = list(range(len(s1)))
        s1, _ = _remove_gaps(s1, lvec1)
        lvec1 = list(range(len(s1)))
        seq_template = torch.tensor([utils.char_to_index[c] for c in s1])
        s2, lvec2 = utils.mutate_sequence(
            seq_template,
            lvec1,
            int(0.3 * len(s1)),
            int(0.2 * len(s1)),
            self.sub_dists,
            utils.amino_distribution,
        )

        s1 = seq_template
        return s1, s2, lvec1, lvec2, idx % len(self.seqs)


class RealisticAliPairGenerator:
    def __init__(self, steps_per_epoch=10000, n_families=1000, len_generated_seqs=128):
        # 10% indel
        self.n_families = n_families
        self.steps_per_epoch = steps_per_epoch
        self.len_generated_seqs = len_generated_seqs
        self.family_templates = None
        self.family_templates = utils.generate_sequences(
            self.n_families, self.len_generated_seqs, utils.amino_distribution
        )
        self.sub_dists = utils.generate_sub_distributions()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        seq_template = utils.generate_sequences(
            1, self.len_generated_seqs, utils.amino_distribution
        ).squeeze()
        # overfit on the _first_ sequence.
        # without any mutations.
        idx = 0
        # generate 10% indel rate, 30% sub rate
        labelvec1 = list(range(len(seq_template)))
        labelvec2 = list(range(len(seq_template)))
        s1, labelvec1 = utils.mutate_sequence(
            seq_template,
            labelvec1,
            int(0.3 * self.len_generated_seqs),
            int(0.1 * self.len_generated_seqs),
            self.sub_dists,
            utils.amino_distribution,
        )
        s2, labelvec2 = utils.mutate_sequence(
            seq_template,
            labelvec2,
            int(0.3 * self.len_generated_seqs),
            int(0.1 * self.len_generated_seqs),
            self.sub_dists,
            utils.amino_distribution,
        )

        return (
            s1.int(),
            s2.int(),
            labelvec1,
            labelvec2,
            idx % len(self.family_templates),
        )


class UniRefGenerator:
    def __init__(self, uniref_file):
        labels, seqs = utils.fasta_from_file(uniref_file)
        pdb.set_trace()
        labels = [label.split()[0] for label in labels]
        # /home/tc229954/data/prefilter/uniprot


class ClusterIterator:
    """
    I'm going to add in alignments here so we can easily look
    at them without grepping or anything.
    """

    def __init__(self, afa_files, min_seq_len, representative_index):

        self.afa_files = afa_files
        self.min_seq_len = min_seq_len
        self.representative_index = representative_index

        self.seed_sequences = []
        self.seed_labels = []
        self.seed_gapped_sequences = []

        self.query_sequences = []
        self.query_labels = []
        self.query_gapped_sequences = []

        # I'm going to keep a record of the original alignments.
        # They will be in two lists:
        # The first will be the cluster representative alignments.
        # the second will be a list with the same sequence order as the unaligned list.

        label_index = 0
        for fasta in afa_files:
            headers, seqs = utils.fasta_from_file(fasta)
            seqs = [s for s in seqs if len(s.replace("-", "")) >= min_seq_len]
            ungapped_seqs = [s.replace("-", "") for s in seqs]
            ungapped_seqs = [s[:min_seq_len] for s in ungapped_seqs]

            if len(ungapped_seqs) > 1:
                self.seed_sequences.append(ungapped_seqs[representative_index])
                self.seed_gapped_sequences.append(seqs[representative_index])
                self.seed_labels.append(label_index)

                self.query_sequences.extend(ungapped_seqs[representative_index + 1 :])
                self.query_gapped_sequences.extend(seqs[representative_index + 1 :])
                self.query_labels.extend(
                    [
                        label_index
                        for _ in range(len(ungapped_seqs[representative_index + 1 :]))
                    ]
                )

                label_index += 1

        self.seed_sequences = [s[:min_seq_len] for s in self.seed_sequences]

    def get_cluster_representatives(self):
        seeds = []
        for seed in self.seed_sequences:
            replacement = []
            for i, c in enumerate(seed):
                if c in ("X", "B", "U", "O", "Z"):
                    replacement.append(
                        utils.amino_alphabet[
                            np.random.randint(0, len(utils.amino_alphabet))
                        ]
                    )
                else:
                    replacement.append(c)
            seeds.append(
                torch.as_tensor([utils.char_to_index[s.upper()] for s in replacement])
            )

        return seeds, self.seed_labels

    def __len__(self):
        return len(self.query_sequences)

    def __getitem__(self, idx):

        qseq = self.query_sequences[idx][: self.min_seq_len]
        label = self.query_labels[idx]
        replacement = []
        for i, c in enumerate(qseq):
            if c in ("X", "B", "U", "O", "Z"):
                replacement.append(
                    utils.amino_alphabet[
                        np.random.randint(0, len(utils.amino_alphabet))
                    ]
                )
            else:
                replacement.append(c)

        seq = torch.as_tensor([utils.char_to_index[i.upper()] for i in replacement])

        return seq, label, self.query_gapped_sequences[idx]


if __name__ == "__main__":

    f = "/home/tc229954/data/prefilter/uniprot/uniref50_subset.fasta"[:2000]

    gen = UniRefGenerator(f)
    print("hello")
