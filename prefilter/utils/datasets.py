# pylint: disable=no-member
import os
import matplotlib.pyplot as plt
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
from prefilter import DECOY_FLAG
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__all__ = [
    "AlignmentGenerator",
    "MLMSwissProtGenerator",
    "SwissProtGenerator",
    "ClusterIterator",
    "ESMEmbeddingGenerator",
    "ClusterIteratorOld",
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

    def __init__(self, fa_file, minlen, training=True):

        self.fa_file = fa_file
        labels, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s for s in seqs if len(s) > minlen]
        self.training = training
        self.sub_dists = utils.generate_correct_substitution_distributions()
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        self.seq_lens = [40, 50, 60, 70]
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
        # now, subsample s2.
        if int(2 * np.random.rand()):
            len_seq = int(self.seq_lens[np.random.randint(0, len(self.seq_lens))])
            begin = np.random.randint(0, len(s2) - len_seq)
            s2 = s2[begin : begin + len_seq]

        return s1, s2, idx % len(self.seqs)

    def __getitem__(self, idx):
        s1, s2, label = self._sample(idx)
        return s1, s2, label


class ClusterIteratorOld:
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
        if self.min_seq_len is None:
            self.min_seq_len = -1
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
        seq = "".join(sanitized)[:1022]

        return seq, label, self.query_gapped_sequences[idx]


class MLMSwissProtGenerator(SwissProtGenerator):
    def __init__(self, fa_file, minlen=256, training=True):
        super(MLMSwissProtGenerator, self).__init__(fa_file, minlen, training)

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
        return_alignments=False,
    ):

        self.train_afa_files = afa_files
        self.min_seq_len = min_seq_len
        self.representative_index = representative_index
        self.n_seq_per_target_family = n_seq_per_target_family
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
                    print(f"removing {file}")
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
                    continue

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
        qali = self.query_alignments[idx]
        label = self.query_labels[idx]
        sanitized = _sanitize_sequence(qseq)
        sanitized = "".join([s for s in sanitized if s != "."])
        if self.return_alignments:
            return sanitized[:1022], qali, label, self.query_sequences[idx]
        else:
            return sanitized[:1022], label, self.query_sequences[idx]


class ESMEmbeddingGenerator:
    """
    Generates pairs of (transformer embedding, sequence).
    Pairs come from swiss-prot.
    """

    def __init__(
        self,
        esm_embedding_path,
        fasta_path,
        convert_to_tensor=False,
        min_seq_len=256,
        n_seqs=30000,
        training=True,
    ):

        self.training = training
        self.convert_to_tensor = convert_to_tensor

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
        if self.convert_to_tensor:
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
