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
from glob import glob
from sys import stdout

import yaml

import prefilter
import prefilter.utils as utils
from prefilter import DECOY_FLAG
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__all__ = [
    "AlignmentGenerator",
    "MLMSwissProtGenerator",
    "SwissProtGenerator",
    "ClusterIterator",
    "ESMEmbeddingGenerator",
]


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


class AlignmentGenerator:
    """
    Grab two sequences from a set of aligned sequences (in an .afa)
    and return them, plus locations where they are not aligned (i.e.
    have gaps).
    """

    def __init__(
        self,
        afa_files,
        apply_substitutions,
        embed_real_within_generated,
        minlen=256,
        training=True,
    ):

        self.afa_files = afa_files
        self.name_to_alignment = {}
        self.training = training
        self.min_seq_len = minlen
        self.apply_substitutions = apply_substitutions
        self.embed_real_within_generated = embed_real_within_generated
        self.sub_dists = utils.generate_correct_substitution_distributions()
        self.sub_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # store as strings for now.
        for afa in self.afa_files:
            _, seqs = utils.fasta_from_file(afa)

            if minlen is not None:
                # remove gaps then calculate length
                seqs = [s for s in seqs if minlen < len(s.replace("-", ""))]

            if len(seqs) > 1:
                self.name_to_alignment[os.path.basename(afa)] = seqs

        self.length = sum([len(v) for v in self.name_to_alignment.values()])
        self.names = list(self.name_to_alignment.keys())

    def __len__(self):
        if self.training:
            return self.length
        else:
            return 10000

    def _sample(self, idx):

        sampled_ali = self.name_to_alignment[
            self.names[idx % len(self.name_to_alignment)]
        ]
        # grab two random families
        i = np.random.randint(0, len(sampled_ali))

        s1 = _sanitize_sequence(sampled_ali[i])[: self.min_seq_len]

        s1 = torch.tensor([utils.char_to_index[c] for c in s1])

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

        return s1, s2, np.arange(len(s1)), np.arange(len(s2))

    def _embed_real_sequence_within_generated(self, idx):
        sampled_ali = self.name_to_alignment[
            self.names[idx % len(self.name_to_alignment)]
        ]
        # grab a random sequence
        i = np.random.randint(0, len(sampled_ali))
        # chop out a length 100 bit of the sequence:
        s1 = [c for c in _sanitize_sequence(sampled_ali[i]) if c != "-"]
        start = np.random.randint(0, self.min_seq_len - 100)
        s1 = torch.tensor([utils.char_to_index[c] for c in s1[start : start + 100]])
        # mutate it a little bit (or a lot!)
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
        # surround it with generated sequence.
        # i need to fix the loss function for this to work.
        gen_s1, gen_s2 = utils.generate_sequences(
            2, self.min_seq_len, utils.amino_distribution
        )
        gen_s1[start : start + 100] = s1
        gen_s2[start : start + 100] = s2
        labelvec = np.arange(len(gen_s1))
        labelvec[:start] = prefilter.MASK_FLAG
        labelvec[start + 100 :] = prefilter.MASK_FLAG
        return gen_s1, gen_s2, labelvec, labelvec

    def __getitem__(self, idx):
        s1, s2, labelvec1, labelvec2 = self._embed_real_sequence_within_generated(idx)
        return s1, s2, torch.as_tensor(labelvec1), torch.as_tensor(labelvec2)

        if idx % 2 == 0 and self.apply_substitutions:
            s1, s2, labelvec1, labelvec2 = self._sample(idx)
            return s1, s2, torch.as_tensor(labelvec1), torch.as_tensor(labelvec2)

        if idx % 5 == 0 and self.embed_real_within_generated:
            # print("embedding real within generated...")
            s1, s2, labelvec1, labelvec2 = self._embed_real_sequence_within_generated(
                idx
            )
            return s1, s2, torch.as_tensor(labelvec1), torch.as_tensor(labelvec2)

        sampled_ali = self.name_to_alignment[
            self.names[idx % len(self.name_to_alignment)]
        ]
        # grab two random families
        i = np.random.randint(0, len(sampled_ali))
        j = np.random.randint(0, len(sampled_ali))

        # ensure we didn't sample the same amino acid
        while i == j:
            j = np.random.randint(0, len(sampled_ali))

        # remove ambiguous/unknown characters
        gapped_sequence_1 = _sanitize_sequence(sampled_ali[i])
        gapped_sequence_2 = _sanitize_sequence(sampled_ali[j])

        # now, get the aligned amino acids by iterating over the sequences as assigning labels to each character
        # remember, we're going to mask unaligned characters out of the loss, so
        # each position in the labelvector that isn't aligned (a gap in either sequence)
        # will have to be some special mask flag.

        labelvec1 = np.arange(len(gapped_sequence_1))
        labelvec2 = np.arange(len(gapped_sequence_2))

        for i, (c1, c2) in enumerate(zip(gapped_sequence_1, gapped_sequence_2)):
            if c1 == "-" and c2 != "-":
                # amino acid at position i exists in sequence 2
                labelvec2[i] = prefilter.MASK_FLAG
                # but does not exist in sequence 2
                labelvec1[i] = prefilter.DROP_FLAG
            elif c1 != "-" and c2 == "-":
                # amino acid at position i exists in sequence 1
                # we want to keep it since it's valid but mask it out eventually
                labelvec1[i] = prefilter.MASK_FLAG
                # but does not exist in sequence 2
                labelvec2[i] = prefilter.DROP_FLAG
            elif c1 == "-" and c2 == "-":
                # neither exist; going to drop them.
                labelvec1[i] = prefilter.DROP_FLAG
                labelvec2[i] = prefilter.DROP_FLAG
            else:
                # both are aligned, continue on.
                continue
        # now, we have to remove the gaps from the sequences to ingest them into the ML algorithm;
        ungapped_sequence_1, labelvec1 = _remove_gaps(gapped_sequence_1, labelvec1)
        ungapped_sequence_2, labelvec2 = _remove_gaps(gapped_sequence_2, labelvec2)
        ungapped_sequence_1 = torch.as_tensor(
            [utils.char_to_index[i] for i in ungapped_sequence_1]
        )
        ungapped_sequence_2 = torch.as_tensor(
            [utils.char_to_index[i] for i in ungapped_sequence_2]
        )

        return (
            ungapped_sequence_1[: self.min_seq_len],
            ungapped_sequence_2[: self.min_seq_len],
            torch.as_tensor(labelvec1)[: self.min_seq_len],
            torch.as_tensor(labelvec2)[: self.min_seq_len],
        )


def _remove_gaps(sequence, labelvec):
    new_sequence = []
    new_labelvec = []
    assert len(sequence) == len(labelvec)
    for s, l in zip(sequence, labelvec):
        if l != prefilter.DROP_FLAG:
            new_labelvec.append(l)
            new_sequence.append(s)
    return new_sequence, new_labelvec


class SwissProtGenerator:
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, apply_indels, minlen=256, training=True):

        self.fa_file = fa_file
        self.apply_indels = apply_indels
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s[1 : minlen + 1] for s in seqs if minlen < len(s)]
        self.training = training
        self.sub_dists = utils.generate_correct_substitution_distributions()
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.indel_probs = [0.01, 0.05, 0.1, 0.15]
        shuffle(self.seqs)

    def __len__(self):
        if self.training:
            print(len(self.seqs))
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

    def __getitem__(self, idx):
        s1, s2, label = self._sample(idx)
        return s1, s2, label


class MLMSwissProtGenerator(SwissProtGenerator):
    def __init__(self, fa_file, apply_indels, minlen=256, training=True):
        super(MLMSwissProtGenerator, self).__init__(
            fa_file, apply_indels, minlen, training
        )

    def __len__(self):
        if self.training:
            return len(self.seqs) // 50
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
        s1 = torch.tensor([utils.char_to_index[c] for c in s1])
        labelvector = s1.clone()
        # mask 15% of each character
        n_mask = torch.randint(len(s1), size=(int(0.15 * len(s1)),))
        # n replace with mask
        # grab 80% to replace with mask (character 21)
        s1[n_mask[: int(0.8 * len(n_mask))]] = 21
        # replace 10% with a random amino acid:
        start = int(0.8 * len(n_mask))
        end = int(0.9 * len(n_mask))
        s1[n_mask[start:end]] = utils.amino_distribution.sample((end - start,))
        return s1, labelvector, idx

    def __getitem__(self, idx):
        s1, labelvector, label = self._sample(idx)
        return s1, labelvector, label


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
    ):

        self.train_afa_files = afa_files
        self.min_seq_len = min_seq_len
        self.representative_index = representative_index
        self.n_seq_per_target_family = n_seq_per_target_family
        self.include_all_families = include_all_families

        self.seed_sequences = []
        self.seed_labels = []

        self.query_sequences = []
        self.query_labels = []
        # dictionary to map train files to validation files.
        self.train_to_valid = {}
        train_files_to_remove = []

        for file in self.train_afa_files:
            # get corresponding validation file;
            valid_file = file.replace(
                "train", "valid_set_subsampled_by_what_hmmer_gets_right"
            )
            if "-1" in os.path.basename(file):
                valid_file = valid_file.replace("-1", "-2") + ".fa"
            else:
                valid_file = valid_file.replace("-2", "-1") + ".fa"

            if os.path.isfile(valid_file):
                self.train_to_valid[file] = valid_file
            else:
                # print(f"Couldn't find valid file for {file}")
                if not include_all_families:
                    train_files_to_remove.append(file)

        for file in train_files_to_remove:
            self.train_afa_files.remove(file)

        self._build_clustered_dataset()

    def _build_clustered_dataset(self):

        label_index = 0

        for train_file in self.train_afa_files:
            # first, grab representative sequences
            _, train_seqs = utils.fasta_from_file(train_file)
            # no gaps here, forget about headers.
            train_seqs = [
                s.replace(".", "")[: self.min_seq_len]
                for s in train_seqs
                if len(s.replace(".", "")) >= self.min_seq_len
            ]

            if not (len(train_seqs)):
                continue

            # shuffle train seqs...?
            self.seed_sequences.extend(train_seqs[: self.n_seq_per_target_family])
            self.seed_labels.extend(
                [label_index] * len(train_seqs[: self.n_seq_per_target_family])
            )

            if train_file in self.train_to_valid:
                # add them into the validation set;
                valid_file = self.train_to_valid[train_file]
                _, valid_seqs = utils.fasta_from_file(valid_file)

                valid_seqs = [
                    s.replace(".", "")[: self.min_seq_len]
                    for s in valid_seqs
                    if len(s.replace(".", "")) >= self.min_seq_len
                ]

                if not (len(valid_seqs)):
                    # print(
                    #     f"No sequence > {self.min_seq_len} in {valid_file}. Skipping."
                    # )
                    continue

                self.query_sequences.extend(valid_seqs)
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
            sanitized = _sanitize_sequence(seed)
            sanitized = [s for s in sanitized if s != "."]
            seeds.append(
                torch.as_tensor([utils.char_to_index[s.upper()] for s in sanitized])
            )

        return seeds, self.seed_labels

    def __len__(self):
        return len(self.query_sequences)

    def __getitem__(self, idx):

        qseq = self.query_sequences[idx]
        label = self.query_labels[idx]
        sanitized = _sanitize_sequence(qseq)
        sanitized = [s for s in sanitized if s != "."]
        seq = torch.as_tensor([utils.char_to_index[i.upper()] for i in sanitized])

        return seq, label, self.query_sequences[idx]


class ESMEmbeddingGenerator:
    """
    Generates pairs of (transformer embedding, sequence).
    Pairs come from swiss-prot.
    """

    def __init__(
        self,
        esm_embedding_path,
        fasta_path,
        min_seq_len=256,
        n_seqs=30000,
        training=True,
    ):

        self.training = training

        self.esm_embeddings = glob(os.path.join(esm_embedding_path, "*.pt"))
        headers, seqs = utils.fasta_from_file(fasta_path)
        # process sequences
        headers_, seqs_ = [], []
        self.esm_embeddings = []
        print(f"Loading {n_seqs} sequences and embeddings.")
        for i, (header, seq) in enumerate(zip(headers, seqs)):
            stdout.write(f"{i/n_seqs:.3f}\r")
            if i == n_seqs:
                break
            esm_file_path = os.path.join(esm_embedding_path, header + ".pt")
            if os.path.isfile(esm_file_path):
                if len(seq) > min_seq_len:
                    headers_.append(header)
                    seqs_.append(seq[:min_seq_len])
                    embed = torch.load(esm_file_path)["representations"][33]
                    self.esm_embeddings.append(embed[:min_seq_len])

        self.seqs = np.asarray(seqs_)

    def _shuf(self):
        idx = np.random.randint(
            0, len(self.esm_embeddings), size=len(self.esm_embeddings)
        )
        self.esm_embeddings = [self.esm_embeddings[i] for i in idx]
        self.seqs = self.seqs[idx]

    def __len__(self):
        if self.training:
            self._shuf()
            return len(self.esm_embeddings)
        else:
            return len(self.esm_embeddings) // 20

    def __getitem__(self, idx):
        if idx % 1000 == 0:
            self._shuf()
        seq = _sanitize_sequence(self.seqs[idx])
        seq = torch.tensor([utils.char_to_index[c] for c in seq])
        embed = self.esm_embeddings[idx]
        return seq, embed, idx


if __name__ == "__main__":
    from glob import glob

    esm = "/home/tc229954/data/prefilter/uniprot/esm1b_uniprot_sprot/"
    uniprot = "/home/tc229954/data/prefilter/uniprot/uniprot_sprot.fasta"
    gen = ESMEmbeddingGenerator(esm, uniprot, 256)
    for ss, e, i in gen:
        pdb.set_trace()
