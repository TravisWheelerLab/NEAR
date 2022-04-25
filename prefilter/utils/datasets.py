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

__all__ = [
    "RealisticAliPairGenerator",
    "SwissProtGenerator",
]


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


if __name__ == "__main__":
    f = "/home/tc229954/data/prefilter/uniprot/uniref50_subset.fasta"

    gen = UniRefGenerator(f)
    print("hello")
