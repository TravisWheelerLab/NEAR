"""Datasets"""
# pylint: disable=no-member
import logging
import os
from random import shuffle

import numpy as np
import torch

from src import utils
from src.datasets import DataModule
from src.utils.gen_utils import generate_string_sequence

logger = logging.getLogger(__name__)

DECOY_FLAG = -1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def sanitize_sequence(sequence):
    """Replaces X U and O in sequences"""
    sanitized = []
    for char in sequence:
        char = char.upper()
        if char in (
            "X",
            "U",
            "O",
        ):  # ambiguous aminos -- replacing them with some other amino from backgorund distribution
            sampled_char = utils.amino_alphabet[utils.amino_distribution.sample().item()]
            sanitized.append(sampled_char)
            logger.debug("Replacing <X, U, O> with %s", sampled_char)
        elif char == "B":  # can be either D or N
            if int(2 * np.random.rand()) == 1:
                sanitized.append("D")
            else:
                sanitized.append("N")
        elif char == "Z":  # can be either E or Q
            if int(2 * np.random.rand()) == 1:
                sanitized.append("E")
            else:
                sanitized.append("Q")
        else:
            sanitized.append(char)

    return "".join(sanitized)

class SwissProtLoader(DataModule):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):

        self.fa_file = fa_file
        _, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s for s in seqs if len(s) >= minlen]
        self.training = training
        self.sub_dists = utils.create_substitution_distribution(62)
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.minlen = minlen
        shuffle(self.seqs)

    def __len__(self):
        """Number of sequences"""
        return len(self.seqs)

    def collate_fn(self):
        """What is returned when fetching batch
        if None then defaults to __getitem__"""
        return None

    def shuffle(self):
        """shuffle"""
        shuffle(self.seqs)

    def __getitem__(self, idx: int):
        """idx: index of sequence to sample"""
        seq1, seq2, label = self._sample(idx)
        return seq1, seq2, label


class SwissProtGenerator(SwissProtLoader):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):
        """Initialize from SwissProtLoader"""
        super().__init__(fa_file, minlen, training)

    def collate_fn(self):
        def pad(batch):
            """
            Pad batches with views as a dim.
            Input: [n_viewsx...]
            :param batch: list of np.ndarrays encoding protein sequences/logos
            :type batch: List[np.ndarray]
            :return: torch.tensor
            :rtype: torch.tensor
            """

            seqs = [b[0] for b in batch]
            mutated_seqs = [b[1] for b in batch]
            labels = [b[2] for b in batch]
            data = seqs + mutated_seqs
            return (
                torch.stack(data),
                torch.as_tensor(labels),
            )

        return pad

    def _sample(self, idx):
        """Sample a sequence and mutate from BLOSUM-62"""
        if not self.training:
            if idx == 0:
                logger.info("shuffling.")
                self.shuffle()

            idx = np.random.randint(0, len(self.seqs))

        seq = self.seqs[idx]
        # subsample sequence;
        if len(seq) != self.minlen:
            start_idx = np.random.randint(0, len(seq) - self.minlen)
        else:
            start_idx = 0
        seq = seq[start_idx : start_idx + self.minlen]

        sequence = sanitize_sequence(seq)

        sequence = torch.as_tensor(
            [utils.amino_char_to_index[c] for c in sequence]
        )  # map amino to int identity

        n_subs = int(  # NOte: we are potentially replacing with the same thing
            len(sequence) * self.sub_probs[np.random.randint(0, len(self.sub_probs))]
        )

        seq2 = utils.mutate_sequence(
            sequence=sequence,
            substitutions=n_subs,
            sub_distributions=self.sub_dists,
        )
        # this creates a fuzzy tensor.
        seq2 = utils.encode_tensor_sequence(seq2)  # 20x256
        return utils.encode_tensor_sequence(sequence), seq2, idx % len(self.seqs)

    def __getitem__(self, idx):
        """Get a sampled and mutated sequence, vectorized"""
        return self._sample(idx)
