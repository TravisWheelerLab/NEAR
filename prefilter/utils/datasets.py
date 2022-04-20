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
    "SequenceDataset",
    "DecoyIterator",
    "RealisticAliPairGenerator",
    "SequenceIterator",
    "RankingIterator",
    "ContrastiveGenerator",
    "ConstrastiveAliGenerator",
    "LogoBatcher",
    "AliPairGenerator",
    "AliEvaluator",
    "NonDiagonalAliPairGenerator",
]


# this could be sped up if i did it vectorized
# but whatever for now
def _compute_soft_label(e_value: float):
    if e_value < 1e-30:
        e_value = 1e-30
    # take care of underflow / inf values in the np.log10
    x = np.clip(np.floor(np.log10(e_value)) * -1, 0, 5) / 5
    return x


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self, fasta_files: Union[str, List[str]], name_to_class_code: Dict[str, int]
    ) -> None:

        if not len(fasta_files):
            raise ValueError("No fasta files found")

        self.fasta_files = fasta_files
        self.name_to_class_code = name_to_class_code

        if not isinstance(self.fasta_files, list):
            self.fasta_files = [self.fasta_files]

    def _make_label_vector(self, labels, len_sequence):
        """
        Create a label vector with len_sequence separate label vectors.
        Each amino acid should have at least one label and up to N.
        :param labels: list of strings
        :type labels: List[str]
        :param len_sequence: length of sequence
        :type len_sequence: int
        :return:
        :rtype:
        """
        y = np.zeros((self.n_classes, len_sequence))
        for labelstring in labels:

            if isinstance(labelstring, list):
                if len(labelstring) == 3:
                    label, begin, end = labelstring
                elif len(labelstring) == 4:
                    label, begin, end, evalue = labelstring
            else:
                label = labelstring
                begin, end = 0, len_sequence

            y[self.name_to_class_code[label], int(float(begin)) : int(float(end))] = 1

        return torch.as_tensor(y)

    def _class_id_vector(self, labels):
        if isinstance(labels, str):
            labels = [labels]
        class_ids = []
        for label in labels:

            if isinstance(label, list) and len(label) != 1:
                label = label[0]

            if label == DECOY_FLAG:
                return []

            class_ids.append(self.name_to_class_code[label])

        return class_ids

    def _make_multi_hot(self, labels):
        y = np.zeros(self.n_classes)
        class_ids = self._class_id_vector(labels)
        for idx in class_ids:
            y[idx] = 1
        return torch.as_tensor(y)

    def _make_distillation_vector(self, labels, e_values):
        y = np.zeros(self.n_classes)
        class_ids = self._class_id_vector(labels)
        for idx, e_value in zip(class_ids, e_values):
            y[idx] = _compute_soft_label(float(e_value))
        return torch.as_tensor(y, dtype=torch.float32, device=torch.device("cpu"))

    def _encoding_func(self, x):
        return utils.encode_protein_as_one_hot_vector(x.upper())

    def _build_dataset(self):
        raise NotImplementedError("Must build the dataset for each custom iterator.")

    def __len__(self):
        raise NotImplementedError("Must specify length for each custom iterator.")

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Must implement __getitem__ for each custom iterator."
        )

    @property
    def n_classes(self):
        return len(self.name_to_class_code)


class SequenceIterator(SequenceDataset):
    """
    Iterates over the sequences in a/the fasta file(s) and encodes them
    for ingestion into an ml algorithm.
    """

    def __init__(
        self,
        fasta_files,
        name_to_class_code,
        evalue_threshold=None,
        distillation_labels=False,
    ):

        super().__init__(fasta_files, name_to_class_code)

        self.sequences_and_labels = []
        self.distillation_labels = distillation_labels
        self.evalue_threshold = evalue_threshold

        self._build_dataset()

    def _build_dataset(self) -> None:
        # TODO: Write rust extension for this.
        for fasta_file in self.fasta_files:
            print(os.path.basename(fasta_file))
            labels, sequences = utils.fasta_from_file(fasta_file)
            for labelstring, sequence in zip(labels, sequences):
                labelset = utils.parse_labels(labelstring)
                if self.evalue_threshold is not None:
                    new_labelset = []
                    for example in labelset:
                        if float(example[-1]) <= self.evalue_threshold:
                            new_labelset.append(example)
                    labelset = new_labelset
                if not len(labelset):
                    raise ValueError(
                        f"Line in {fasta_file} does not contain any labels. Please fix."
                    )

                if len(labelset) == 1:
                    if labelset[0] == "DECOY":
                        # no labels for decoys
                        lvec = torch.as_tensor(np.zeros(self.n_classes))
                    elif len(labelset[0]) != 4:
                        # it's an emission sequence
                        lvec = self._make_distillation_vector(labelset, [1e-30])
                    else:
                        # its eeal but only has one label
                        lvec = self._make_distillation_vector(
                            [l[0] for l in labelset], [l[-1] for l in labelset]
                        )
                else:
                    lvec = self._make_distillation_vector(
                        [l[0] for l in labelset], [l[-1] for l in labelset]
                    )

                self.sequences_and_labels.append([sequence, lvec])

    def __getitem__(self, idx):
        features, labels = self.sequences_and_labels[idx]
        encoded_features = self._encoding_func(features)
        return torch.as_tensor(encoded_features), labels

    def __len__(self):
        return len(self.sequences_and_labels)

    def shuffle(self):
        shuffle(self.sequences_and_labels)


class DecoyIterator(SequenceDataset):
    """
    Iterates over label-less fasta files (decoys, usually).
    """

    def __init__(self, fasta_files, name_to_class_code):
        super().__init__(fasta_files, name_to_class_code)

        self.sequences = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        for fasta_file in self.fasta_files:
            _, sequences = utils.fasta_from_file(fasta_file)
            self.sequences.extend(sequences)

    def __getitem__(self, idx):
        example = self.sequences[idx]
        return self._encoding_func(example[0]), torch.zeros(self.n_classes)

    def __len__(self):
        return len(self.sequences)


class RankingIterator(SequenceDataset):
    def __init__(
        self, fasta_files, name_to_class_code, max_labels_per_seq, evalue_threshold
    ):
        super().__init__(fasta_files, name_to_class_code)

        self._build_dataset(max_labels_per_seq, evalue_threshold)

    def _build_dataset(self, max_labels_per_seq, evalue_threshold):
        self.labels_and_sequences = utils.load_sequences_and_labels(
            self.fasta_files, max_labels_per_seq, evalue_threshold
        )

    def __getitem__(self, idx):
        labelset, sequence = self.labels_and_sequences[idx]
        return self._encoding_func(sequence), self._make_multi_hot(labelset), labelset

    def __len__(self):
        return len(self.labels_and_sequences)


class SequenceIterator(SequenceDataset):
    def __init__(
        self, fasta_files, name_to_class_code, max_labels_per_seq, evalue_threshold
    ):
        super().__init__(fasta_files, name_to_class_code)

        self._build_dataset(max_labels_per_seq, evalue_threshold)

    def _build_dataset(self, max_labels_per_seq, evalue_threshold):
        self.labels_and_sequences = utils.load_sequences_and_labels(
            self.fasta_files, max_labels_per_seq, evalue_threshold
        )

    def __getitem__(self, idx):
        labelset, sequence = self.labels_and_sequences[idx]
        return self._encoding_func(sequence), labelset

    def __len__(self):
        return len(self.labels_and_sequences)


def _remove_gaps(seq, label):
    _seq = []
    _label = []
    for s, l in zip(seq, label):
        if s not in ("-", "."):
            _seq.append(s)
            _label.append(l)
    return _seq, _label


class ConstrastiveAliGenerator:
    def __init__(self, afa_files, pad=True, length_of_seq=None):

        self.afa_files = afa_files
        self.length_of_seq = length_of_seq
        self.sub_dists = utils.generate_sub_distributions()
        self.pad = pad

        self._build_dataset()

    def _build_dataset(self):
        self.alidb = []
        for afa in self.afa_files:
            seqs = utils.afa_from_file(afa)
            # if self.length_of_seq is not None and not self.pad:
            #     seqs = [s[:self.length_of_seq] for s in seqs if len(s) > self.length_of_seq]
            # elif self.length_of_seq is not None and self.pad:
            #     seqs = [s[:self.length_of_seq] for s in seqs]
            if len(seqs) > 3:
                self.alidb.append(seqs)
        self.len = sum(list(map(len, self.alidb)))

    def __len__(self):
        return self.len

    def _sample(self, idx):
        ali = self.alidb[idx % len(self.alidb)]
        i = np.random.randint(0, len(ali))

        s1 = [x.upper() for x in ali[i] if x not in ("X", "B", "U")]

        lvec1 = list(range(len(s1)))
        s1, _ = _remove_gaps(s1, lvec1)
        if len(s1) < self.length_of_seq and self.pad:
            pad_len = self.length_of_seq - len(s1)
            if pad_len == 1:
                random_seq = utils.generate_sequences(
                    1, pad_len, utils.amino_distribution
                )
            else:
                random_seq = utils.generate_sequences(
                    1, pad_len, utils.amino_distribution
                ).squeeze()
            s1 = s1 + [utils.amino_alphabet[c.item()] for c in random_seq]

        lvec1 = list(range(len(s1)))
        seq_template = torch.tensor([utils.char_to_index[c] for c in s1])
        s2, lvec2 = utils.mutate_sequence(
            seq_template,
            lvec1,
            int(0.3 * len(s1)),
            int(0.25 * len(s1)),
            self.sub_dists,
            utils.amino_distribution,
        )

        s2 = utils.encode_protein_as_one_hot_vector(
            "".join([utils.amino_alphabet[c.item()] for c in s2])
        )
        s1 = utils.encode_protein_as_one_hot_vector("".join(s1))

        return s1, s2, lvec1, lvec2, idx % len(self.alidb)

    def __getitem__(self, idx):
        return self._sample(idx)


class AliEvaluator(ConstrastiveAliGenerator):
    def __init__(self, afa_files, length_of_seq=None, pad=True):
        super().__init__(afa_files, length_of_seq=length_of_seq, pad=pad)
        self.seed_seqs = True
        self.len = sum([len(l[1:]) for l in self.alidb])
        self.flattened = []
        for i, ali in enumerate(self.alidb):
            for seq in ali[1:]:
                self.flattened.append([i, seq])

    def __len__(self):
        if self.seed_seqs:
            return len(self.alidb)
        else:
            return self.len

    def __getitem__(self, idx):
        if self.seed_seqs:
            seq = "".join(self.alidb[idx][0])
            if len(seq) < 120 and self.pad:
                random_seq = utils.generate_sequences(
                    1, 120 - len(seq), utils.amino_distribution
                ).squeeze()
                seq = seq + [utils.amino_alphabet[c.item()] for c in random_seq]
            return utils.encode_protein_as_one_hot_vector(seq), idx
        else:
            label, seq = self.flattened[idx]
            seq = [s.upper() for s in seq]
            if self.length_of_seq is not None:
                if len(seq) < self.length_of_seq and self.pad:
                    pad_len = self.length_of_seq - len(seq)
                    if pad_len == 1:
                        random_seq = utils.generate_sequences(
                            1, pad_len, utils.amino_distribution
                        )
                    else:
                        random_seq = utils.generate_sequences(
                            1, pad_len, utils.amino_distribution
                        ).squeeze()
                    seq = seq + [utils.amino_alphabet[c.item()] for c in random_seq]
            seq = "".join(seq)
            return utils.encode_protein_as_one_hot_vector(seq), label


class ContrastiveGenerator(SequenceDataset):
    def __init__(
        self,
        fasta_files,
        logo_path,
        name_to_class_code,
        oversample_neighborhood_labels,
        oversample_freq=5,
    ):

        if not len(fasta_files):
            raise ValueError("No fasta files found")

        self.fasta_files = fasta_files
        self.logo_path = logo_path
        self.name_to_sequences = defaultdict(list)
        self.name_to_class_code = name_to_class_code
        self.oversample_freq = oversample_freq
        self.name_to_logo = {}
        self.names = None
        self.oversample_neighborhood_labels = oversample_neighborhood_labels
        if self.oversample_neighborhood_labels:
            self.neighborhood_name_to_sequences = defaultdict(list)
            self.neighborhood_names = None
            self.step_count = 0

        if not isinstance(self.fasta_files, list):
            self.fasta_files = [self.fasta_files]

        self._build_dataset()

    def _build_dataset(self):

        with open(prefilter.name_to_accession_id, "r") as src:
            name_to_acc_id = yaml.safe_load(src)

        acc_id_to_name = {v: k for k, v in name_to_acc_id.items()}
        del name_to_acc_id

        for fasta_file in self.fasta_files:
            print(f"loading sequences from {fasta_file}")
            labels, sequences = utils.fasta_from_file(fasta_file)

            if "emission" in os.path.basename(fasta_file):
                print("truncating...", len(sequences))
                labels = labels[:20]
                sequences = sequences[:20]

            for labelstring, sequence in zip(labels, sequences):
                # parse labels
                labelset = utils.parse_labels(labelstring)
                labelset = list(filter(lambda x: float(x[-1]) < 1e-5, labelset))
                labelset = [lab[0] for lab in labelset]

                for k, label in enumerate(labelset):
                    if self.oversample_neighborhood_labels:
                        if k == 0:
                            self.name_to_sequences[label].append(
                                utils.encode_protein_as_one_hot_vector(sequence)
                            )
                        else:
                            self.neighborhood_name_to_sequences[label].append(
                                utils.encode_protein_as_one_hot_vector(sequence)
                            )
                    else:
                        self.name_to_sequences[label].append(
                            utils.encode_protein_as_one_hot_vector(sequence)
                        )
        # now construct the name-to-logo by
        # iterating over the pfam accession IDs in name to sequences
        if self.oversample_neighborhood_labels:
            accession_ids = set(
                list(self.name_to_sequences.keys())
                + list(self.neighborhood_name_to_sequences.keys())
            )
        else:
            accession_ids = list(self.name_to_sequences.keys())

        for accession_id in accession_ids:
            logo_name = acc_id_to_name[accession_id]
            logo_file = os.path.join(self.logo_path, f"{logo_name}.0.5-train.hmm.logo")
            if not os.path.isfile(logo_file):
                raise ValueError(f"No logo file found at {logo_file}")
            else:
                self.name_to_logo[accession_id] = utils.logo_from_file(logo_file)

        if self.oversample_neighborhood_labels:
            self.names = list(self.name_to_sequences.keys())
            self.neighborhood_names = list(self.neighborhood_name_to_sequences.keys())
        else:
            self.names = list(self.name_to_sequences.keys())

        self.len = sum(list(map(len, list(self.name_to_sequences.values()))))

    def __len__(self):
        return self.len

    def _sample(self, name_to_sequences, names):
        """
        Helper function to sample from a dict mapping pfam accession id to a list of sequences
        :param dct:
        :type dct:
        :return:
        :rtype:
        """
        name_idx = int(np.random.rand() * len(names))
        name = names[name_idx]
        pos_seqs, pos_logo = name_to_sequences[name], self.name_to_logo[name]
        seq_idx = int(np.random.rand() * len(pos_seqs))
        pos_seq = pos_seqs[seq_idx]
        return pos_seq, pos_logo, self.name_to_class_code[name]

    def __getitem__(self, idx):
        # stochastic
        if self.oversample_neighborhood_labels:
            if self.step_count % self.oversample_freq == 0:
                pos_seq, pos_logo, class_code = self._sample(
                    self.name_to_sequences, self.names
                )
            else:
                pos_seq, pos_logo, class_code = self._sample(
                    self.neighborhood_name_to_sequences, self.neighborhood_names
                )
            self.step_count += 1
        else:
            pos_seq, pos_logo, class_code = self._sample(
                self.name_to_sequences, self.names
            )

        return pos_seq, pos_logo, class_code


class LogoBatcher(SequenceDataset):
    def __init__(self, fasta_files, name_to_class_code):
        super().__init__(fasta_files, name_to_class_code)
        self.logo_files = fasta_files
        self.logos = []

        self._build_dataset()

    def _build_dataset(self):
        for logo_file in self.logo_files:
            self.logos.append(utils.logo_from_file(logo_file))

    def __len__(self):
        return len(self.logo_files)

    def __getitem__(self, idx):
        return self.logos[idx], 0


class AliPairGenerator:
    def __init__(self, steps_per_epoch=10000, len_generated_seqs=100, num_seeds=1000):

        np.random.seed(10)

        self.steps_per_epoch = steps_per_epoch
        self.len_generated = len_generated_seqs
        self.alphabet = list(utils.PROT_ALPHABET.keys())
        self.mapping = utils.PROT_ALPHABET
        self.run_length = 10
        self.num_seeds = num_seeds

        self._build_dataset(num_seeds)

    def _choose_aa(self):
        return self.alphabet[int(np.random.rand() * len(self.alphabet))]

    def _gen_normalized_pvec(self, conserved_aa, aa_prob):
        random_probs = np.random.rand(len(self.alphabet))
        random_probs[self.mapping[conserved_aa]] = 0
        random_probs = random_probs / np.sum(random_probs)
        random_probs *= 1 - aa_prob
        random_probs[self.mapping[conserved_aa]] = aa_prob
        return random_probs

    def _build_dataset(self, num_seeds):
        n_highly_conserved = int(self.len_generated * 0.2)
        # i want some highly conserved AAs, others not really.
        # let's say I want 20 highly conserved AAs.
        # And 40 less-conserved AAs.
        self.seed_list = []
        probs = [0.6, 0.7, 0.8, 0.9, 0.95]
        for _ in range(num_seeds):
            # max here will be 0.5.
            generated_template = np.zeros((self.len_generated, len(self.alphabet)))
            i = 0
            while i < self.len_generated:
                # draw
                draw = int(np.random.rand() * self.len_generated)
                if draw <= n_highly_conserved:
                    # regions of nicely conserved amino acids
                    random_conserved_aa = self._choose_aa()
                    prob_choice = probs[int(np.random.rand() * len(probs))]
                    generated_template[i] = self._gen_normalized_pvec(
                        random_conserved_aa, prob_choice
                    )
                    j = 0
                    i += 1
                    while j < self.run_length and i < self.len_generated:
                        random_conserved_aa = self._choose_aa()
                        prob_choice = probs[int(np.random.rand() * len(probs))]
                        generated_template[i] = self._gen_normalized_pvec(
                            random_conserved_aa, prob_choice
                        )

                        i += 1
                        j += 1
                else:
                    # random density.
                    rvec = np.random.rand(len(self.alphabet))
                    generated_template[i] = rvec / np.sum(rvec)
                    i += 1

            self.seed_list.append(generated_template)

    def _generate_seq(self, seq_template):
        seq = []
        for i in range(len(seq_template)):
            seq.append(np.random.choice(self.alphabet, p=seq_template[i]))
        return utils.encode_protein_as_one_hot_vector("".join(seq))

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        return (
            self._generate_seq(self.seed_list[idx % len(self.seed_list)]),
            self._generate_seq(self.seed_list[idx % len(self.seed_list)]),
            idx % len(self.seed_list),
        )


class NonDiagonalAliPairGenerator(AliPairGenerator):
    def __init__(self, steps_per_epoch=100000, len_generated_seqs=100, num_seeds=1000):
        super().__init__(steps_per_epoch, len_generated_seqs, num_seeds)
        # Do I care about not mutating highly conserved AAs?
        # probably... or else the classifier will freak out
        self.num_inserts_or_deletions = [5, 6, 7, 8, 9]
        self.insert_or_del_run_length = [4, 5, 6, 7, 8]

    def _choose(self, arr):
        """
        Grab a random element from an array (not using np.random.choice!)
        """
        return arr[np.random.randint(0, len(arr))]

    def mutate(self, sequence, lvec, lvec_start):
        """
        Randomly insert or delete AAs from a sequence in runs.
        Random start positions are chosen N times for both insertions and deletions.
        """
        for _ in range(self._choose(self.num_inserts_or_deletions)):
            pos = np.random.randint(0, len(sequence))
            run_length = self._choose(self.insert_or_del_run_length)
            # insertion step
            for j in range(pos, min(pos + run_length, len(sequence))):
                random_aa = self.alphabet[np.random.randint(0, len(self.alphabet))]
                sequence.insert(pos + j, random_aa)
                # add unique label
                lvec.insert(pos + j, lvec_start)
                lvec_start += 1

        for _ in range(self._choose(self.num_inserts_or_deletions)):
            # del step
            pos = np.random.randint(0, len(sequence))
            run_length = self._choose(self.insert_or_del_run_length)
            # insertion step
            for j in range(pos, min(pos + run_length, len(sequence))):
                sequence.pop(pos)
                lvec.pop(pos)

        return sequence, lvec

    def _generate_seq(self, seq_template):
        seq = []
        for i in range(len(seq_template)):
            seq.append(np.random.choice(self.alphabet, p=seq_template[i]))
        return seq

    def __getitem__(self, idx):
        s1 = self._generate_seq(self.seed_list[idx % len(self.seed_list)])
        # lists for easy insertion
        lvec1 = list(range(len(s1)))
        s1, lvec1 = self.mutate(s1, lvec1, len(lvec1) + 1)
        s2 = self._generate_seq(self.seed_list[idx % len(self.seed_list)])
        lvec2 = list(range(len(s2)))
        s2, lvec2 = self.mutate(s2, lvec2, np.max(lvec1) + 1)
        s1 = utils.encode_protein_as_one_hot_vector("".join(s1))
        s2 = utils.encode_protein_as_one_hot_vector("".join(s2))
        return s1, s2, np.asarray(lvec1), np.asarray(lvec2), idx % len(self.seed_list)


class RealisticAliPairGenerator:
    def __init__(self, steps_per_epoch=10000, n_families=1000, len_generated_seqs=256):
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

        s1 = "".join([utils.amino_alphabet[c.item()] for c in s1])
        s2 = "".join([utils.amino_alphabet[c.item()] for c in s2])
        return (
            utils.encode_protein_as_one_hot_vector(s1),
            utils.encode_protein_as_one_hot_vector(s2),
            labelvec1,
            labelvec2,
            idx % len(self.family_templates),
        )


if __name__ == "__main__":
    from glob import glob

    train = glob(
        "/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/*-train.sto.afa"
    )[:10]
    gen = ConstrastiveAliGenerator(afa_files=train)

    for s1, s2, l1, l2, l, m in gen:
        pdb.set_trace()
