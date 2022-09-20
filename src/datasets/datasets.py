# pylint: disable=no-member
import json
import os
import pdb
import string
import time
import warnings
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from glob import glob
from random import shuffle
from sys import stdout
from typing import Dict, List, Optional, Tuple, Union

import esm as esm
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import src
import src.models as models
import src.utils as utils
from src.datasets import DataModule

DECOY_FLAG = -1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__all__ = ["SwissProtGenerator", "ClusterIterator", "ProfmarkDataset"]


def sanitize_sequence(sequence):

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


class SwissProtGenerator(DataModule):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):

        self.fa_file = fa_file
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s[:minlen] for s in seqs if len(s) >= minlen]
        self.training = training
        self.sub_dists = utils.create_substituion_distribution(62)
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        shuffle(self.seqs)

    def __len__(self):
        if self.training:
            return len(self.seqs) // 10
        else:
            return 10000

    def collate_fn(self):
        return utils.pad_contrastive_batches

    def shuffle(self):
        shuffle(self.seqs)

    def _sample(self, idx):
        if not self.training:
            if idx == 0:
                print("shuffling.")
                self.shuffle()

            idx = np.random.randint(0, len(self.seqs))

        s1 = sanitize_sequence(self.seqs[idx])

        s1 = torch.as_tensor([utils.amino_char_to_index[c] for c in s1])
        n_subs = int(
            len(s1) * self.sub_probs[np.random.randint(0, len(self.sub_probs))]
        )
        s2 = utils.mutate_sequence(
            sequence=s1, substitutions=n_subs, sub_distributions=self.sub_dists
        )
        return s1, s2, idx % len(self.seqs)

    def __getitem__(self, idx):
        s1, s2, label = self._sample(idx)
        return s1, s2, label


class SwissProtGeneratorDanielSequenceEncode(SwissProtGenerator):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):
        super().__init__(fa_file, minlen, training)

    def _sample(self, idx):
        if not self.training:
            if idx == 0:
                print("shuffling.")
                self.shuffle()

            idx = np.random.randint(0, len(self.seqs))

        sequence = sanitize_sequence(self.seqs[idx])
        # ok, it's fine if it's different.

        sequence = torch.as_tensor([utils.amino_char_to_index[c] for c in sequence])

        n_subs = int(
            len(sequence) * self.sub_probs[np.random.randint(0, len(self.sub_probs))]
        )

        s2 = utils.mutate_sequence(
            sequence=sequence,
            substitutions=n_subs,
            sub_distributions=self.sub_dists,
        )
        # this creates a fuzzy tensor.
        s2 = utils.encode_tensor_sequence(s2)
        return utils.encode_tensor_sequence(sequence), s2, idx % len(self.seqs)

    def __getitem__(self, idx):
        return self._sample(idx)

    def collate_fn(self):
        return utils.pad_contrastive_batches
