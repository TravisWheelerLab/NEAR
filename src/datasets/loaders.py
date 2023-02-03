# pylint: disable=no-member
import logging
import os
from glob import glob
from random import shuffle

import numpy as np
import torch
from Bio import AlignIO
from torchaudio.transforms import MelSpectrogram

import src.utils as utils
from src.datasets import DataModule
from src.datasets.dataset import sanitize_sequence
from src.utils.gen_utils import generate_string_sequence
from src.utils.helpers import AAIndexFFT

logger = logging.getLogger(__name__)

DECOY_FLAG = -1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class SwissProtLoaderGeneralInput(DataModule):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(
        self, minlen, labels: list = None, seqs: list = None, fa_file: str = None, training=True,
    ):

        if fa_file:
            self.fa_file = fa_file
            labels, seqs = utils.fasta_from_file(fa_file)
        else:
            assert (
                labels is not None and seqs is not None
            ), "You must input an fa file or labels and sequences"
        self.seqs = [s for s in seqs if len(s) >= minlen]
        self.training = training
        self.sub_dists = utils.create_substitution_distribution(62)
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.minlen = minlen
        shuffle(self.seqs)

    def __len__(self):
        return len(self.seqs)

    def collate_fn(self):
        return None

    def shuffle(self):
        shuffle(self.seqs)

    def __getitem__(self, idx):
        s1, s2, label = self._sample(idx)
        return s1, s2, label


class SwissProtGenerator(SwissProtLoaderGeneralInput):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, minlen, labels=None, seqs=None, fa_file=None, training=True):
        super(SwissProtGenerator, self).__init__(minlen, labels, seqs, fa_file, training)

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

        s2 = utils.mutate_sequence(
            sequence=sequence, substitutions=n_subs, sub_distributions=self.sub_dists,
        )
        # this creates a fuzzy tensor.
        s2 = utils.encode_tensor_sequence(s2)  # 20x256
        return utils.encode_tensor_sequence(sequence), s2, idx % len(self.seqs)

    def __getitem__(self, idx):
        return self._sample(idx)
