# pylint: disable=no-member
import logging
import os
from glob import glob
from random import shuffle

import numpy as np
import torch
from Bio import AlignIO

import src.utils as utils
from src.datasets import DataModule
from src.utils.gen_utils import generate_string_sequence

logger = logging.getLogger(__name__)

DECOY_FLAG = -1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def sanitize_sequence(sequence):

    sanitized = []
    for char in sequence:
        char = char.upper()
        if char in ("X", "U", "O"):
            sampled_char = utils.amino_alphabet[
                utils.amino_distribution.sample().item()
            ]
            sanitized.append(sampled_char)
            logger.debug(f"Replacing <X, U, O> with {sampled_char}")
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

    return "".join(sanitized)


class SequenceIterator(DataModule):
    def __init__(self, fa_file, length):

        self.fa_file = fa_file
        labels, seqs = utils.fasta_from_file(fa_file)
        self.labels = []
        self.seqs = []
        for label, seq in zip(labels, seqs):
            if len(seq) >= length:
                self.labels.append(label)
                self.seqs.append(seq[:length])

    def __len__(self):
        return len(self.seqs)

    def collate_fn(self):
        return utils.pad_contrastive_batches_daniel

    def __getitem__(self, idx):
        s1 = sanitize_sequence(self.seqs[idx])
        return utils.encode_string_sequence(s1), self.labels[idx]


class AlignmentGenerator(DataModule):
    def collate_fn(self):
        return None

    def __init__(self, ali_path, seq_len, training=True):
        self.alignment_files = glob(os.path.join(ali_path, "*.sto"))
        # also put a classification loss on there.
        # so i can see accuracy numbers
        self.training = training
        if len(self.alignment_files) == 0:
            raise ValueError(f"No alignment files found at {ali_path}.")
        self.mx = 0
        self.seq_len = seq_len

    def __len__(self):
        if self.training:
            return 10000
        else:
            return 500

    def __getitem__(self, idx):
        ali = AlignIO.read(self.alignment_files[idx], "stockholm")._records
        # aligned index with gaps: aligned index without gaps
        seq1 = ali[0].upper()
        seq2 = ali[1].upper()
        seq1 = sanitize_sequence(seq1)
        seq2 = sanitize_sequence(seq2)
        seq_a_aligned_labels = list(range(self.mx, self.mx + len(seq1)))
        seq_b_aligned_labels = list(range(self.mx, self.mx + len(seq2)))
        # just remove elements from the labels above
        # that are gap characters in either sequence.
        # then labels that are the same will be aligned characters
        seq_a_to_keep = []
        for j, amino in enumerate(seq1):
            if amino not in ("-", "."):
                seq_a_to_keep.append(j)

        seq_a_aligned_labels = [seq_a_aligned_labels[s] for s in seq_a_to_keep]
        seq_b_to_keep = []
        for j, amino in enumerate(seq2):
            if amino not in ("-", "."):
                seq_b_to_keep.append(j)

        seq_b_aligned_labels = [seq_b_aligned_labels[s] for s in seq_b_to_keep]
        seq1 = seq1.replace(".", "").replace("-", "")
        seq2 = seq2.replace(".", "").replace("-", "")

        seq1_chop = len(seq1) - self.seq_len
        seq2_chop = len(seq2) - self.seq_len

        if seq1_chop > 0:
            seq1 = seq1[seq1_chop // 2 : -seq1_chop // 2]
            seq_a_aligned_labels = seq_a_aligned_labels[
                seq1_chop // 2 : -seq1_chop // 2
            ]
        elif seq1_chop == 0:
            pass
        else:
            # add characters to the front
            addition = generate_string_sequence(-seq1_chop)
            # now add bullshit to the labels at the beginning
            mx = max(max(seq_a_aligned_labels), max(seq_b_aligned_labels)) + 1
            seq1 = addition + seq1
            for _ in range(len(addition)):
                seq_a_aligned_labels.insert(0, mx)
                mx += 1

        if seq2_chop > 0:
            seq2 = seq2[seq2_chop // 2 : -seq2_chop // 2]
            seq_b_aligned_labels = seq_b_aligned_labels[
                seq2_chop // 2 : -seq2_chop // 2
            ]
        elif seq2_chop == 0:
            pass
        else:
            # add characters to the front
            addition = generate_string_sequence(-seq2_chop)
            # now add bullshit to the labels at the beginning
            mx = max(max(seq_a_aligned_labels), max(seq_b_aligned_labels)) + 1
            seq2 = addition + seq2
            for _ in range(len(addition)):
                seq_b_aligned_labels.insert(0, mx)
                mx += 1
        # brutally chop off the ends?
        self.mx = max(max(seq_a_aligned_labels), max(seq_b_aligned_labels)) + 1
        return (
            utils.encode_string_sequence(seq1),
            torch.as_tensor(seq_a_aligned_labels),
            utils.encode_string_sequence(seq2),
            torch.as_tensor(seq_b_aligned_labels),
        )


class SwissProtGenerator(DataModule):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):

        self.fa_file = fa_file
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s for s in seqs if len(s) >= minlen]
        self.training = training
        self.sub_dists = utils.create_substitution_distribution(62)
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.minlen = minlen
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

        seq = self.seqs[idx]
        # subsample sequence;
        start_idx = np.random.randint(0, len(seq) - self.minlen)
        seq = seq[start_idx : start_idx + self.minlen]

        s1 = sanitize_sequence(seq)

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
        return utils.stack_contrastive_batch


class FastaSampler:
    def __init__(self, train_fasta, valid_fasta):
        _, self.train_sequences = utils.fasta_from_file(train_fasta)
        _, self.valid_sequences = utils.fasta_from_file(valid_fasta)

    def sample(self):
        # grab a random pair
        train_idx = int(np.random.rand() * len(self.train_sequences))
        valid_idx = int(np.random.rand() * len(self.valid_sequences))
        return self.train_sequences[train_idx], self.valid_sequences[valid_idx]


class PfamDataset(DataModule):
    def __init__(self, train_files, valid_files, training=True, **kwargs):

        super(PfamDataset, self).__init__(**kwargs)

        self.training = training

        if len(train_files) == 0 or len(valid_files) == 0:
            raise ValueError("Didn't receive any train/valid files.")

        self.training_pairs = []
        for i, train in enumerate(train_files):
            valid_name = train.replace("-train", "-valid")
            if valid_name in valid_files:
                valid_file = valid_files[valid_files.index(valid_name)]
                self.training_pairs.append((i, FastaSampler(train, valid_file)))
            else:
                self.training_pairs.append((i, FastaSampler(train, train)))

    def collate_fn(self):
        def pad_view_batches(batch):
            """
            Pad batches with views as a dim.
            Input: [n_viewsx...]
            :param batch: list of np.ndarrays encoding protein sequences/logos
            :type batch: List[np.ndarray]
            :return: torch.tensor
            :rtype: torch.tensor
            """

            seqs = [b[0] for b in batch]
            logos = [b[1] for b in batch]
            labels = [b[2] for b in batch]
            data = seqs + logos
            seqs, seqs_mask = utils.pad_sequences(data)
            return (
                torch.as_tensor(seqs),
                torch.as_tensor(seqs_mask),
                torch.as_tensor(labels),
            )

        return pad_view_batches

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        label, sampler = self.training_pairs[index // len(self.training_pairs)]
        s1, s2 = sampler.sample()
        if len(s1) < 128:
            s1 = s1 + "".join(["A"] * 128)
        # same size sequences, will this fit?
        s1 = s1[:128]
        return (
            utils.encode_string_sequence(sanitize_sequence(s1)),
            utils.encode_string_sequence(sanitize_sequence(s1)),
            label,
        )
