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
from src.utils.gen_utils import generate_string_sequence
from src.utils.helpers import AAIndexFFT

logger = logging.getLogger(__name__)

DECOY_FLAG = -1

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def sanitize_sequence(sequence):
    # TODO: can make this faster (do it once with data)
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
            logger.debug(f"Replacing <X, U, O> with {sampled_char}")
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
        from Bio import AlignIO

        self.alignment_files = glob(os.path.join(ali_path, "*.sto"))
        self.training = training

        if self.training:
            self.alignment_files = self.alignment_files[:-100]
        else:
            self.alignment_files = self.alignment_files[:100]
        # also put a classification loss on there.
        # so i can see accuracy numbers
        if len(self.alignment_files) == 0:
            raise ValueError(f"No alignment files found at {ali_path}.")
        self.mx = 0
        self.seq_len = seq_len

    def __len__(self):
        return len(self.alignment_files)

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
            seq_a_aligned_labels = seq_a_aligned_labels[seq1_chop // 2 : -seq1_chop // 2]
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
            seq_b_aligned_labels = seq_b_aligned_labels[seq2_chop // 2 : -seq2_chop // 2]
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


class SwissProtLoader(DataModule):
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
        return len(self.seqs)

    def collate_fn(self):
        return None

    def shuffle(self):
        shuffle(self.seqs)

    def __getitem__(self, idx):
        s1, s2, label = self._sample(idx)
        return s1, s2, label


class SwissProtGenerator(SwissProtLoader):
    """
    Grab a sequence from swiss-prot, mutate it, then
    feed it to the model as a contrastive pair.
    """

    def __init__(self, fa_file, minlen, training=True):
        super(SwissProtGenerator, self).__init__(fa_file, minlen, training)

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
            return (
                torch.stack(seqs + logos),
                torch.as_tensor(labels),
            )

        return pad_view_batches

    def __len__(self):
        if self.training:
            return 100000
        else:
            return 10000

    def __getitem__(self, index):
        label, sampler = self.training_pairs[index // len(self.training_pairs)]
        s1, s2 = sampler.sample()

        while len(s1) < 128:
            s1 = s1 + s1
        while len(s2) < 128:
            s2 = s2 + s2
        # same size sequences, will this fit?
        s1 = s1[:128]
        s2 = s2[:128]
        return (
            utils.encode_string_sequence(sanitize_sequence(s1)),
            utils.encode_string_sequence(sanitize_sequence(s2)),
            label,
        )


class KMerSampler(DataModule):
    def __init__(self, training, min_kmer_length, max_kmer_length, minlen=32):
        super(KMerSampler, self).__init__()
        self.training = training
        self.minlen = minlen
        self.max_kmer_length = max_kmer_length
        self.min_kmer_length = min_kmer_length
        self.idx_cnt = 0

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
            return (
                torch.stack(data),
                torch.as_tensor(labels),
            )

        return pad_view_batches

    def __len__(self) -> int:
        if self.training:
            return 200000
        else:
            return 10000

    def __getitem__(self, index: int):
        # make a new sequence:
        seed_sequence = generate_string_sequence(self.minlen)
        # grab a random kmer from the seed sequence
        kmer_length = np.random.randint(
            low=self.min_kmer_length, high=self.max_kmer_length, size=1
        )[0]
        # kmer_length = max(kmer_length, 5)
        start_idx = int(np.random.rand() * (self.minlen - kmer_length))
        # make a kmer seed
        kmer_seed = seed_sequence[start_idx : start_idx + kmer_length]
        random_seq = generate_string_sequence(self.minlen)
        start_idx = int(np.random.rand() * (self.minlen - kmer_length))
        seeded_seq = random_seq[:start_idx] + kmer_seed + random_seq[start_idx + kmer_length :]
        self.idx_cnt += 1
        # just do pairs for now.
        return (
            utils.encode_string_sequence(sanitize_sequence(seed_sequence)),
            utils.encode_string_sequence(sanitize_sequence(seeded_seq)),
            self.idx_cnt,
        )


class SpectrogramDataset(DataModule):
    def __init__(self, training, minlen, kmer_length):
        super(SpectrogramDataset, self).__init__()
        self.mel = MelSpectrogram(win_length=32, hop_length=16)
        self.training = training
        self.minlen = minlen
        self.kmer_length = kmer_length
        self.idx_cnt = 0

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
            return (
                torch.stack(data),
                torch.as_tensor(labels),
            )

        return pad_view_batches

    def __len__(self) -> int:
        if self.training:
            return 200000
        else:
            return 10000

    def __getitem__(self, index: int):
        # make a new sequence:
        seed_sequence = generate_string_sequence(self.minlen)
        # grab a random kmer from the seed sequence
        kmer_length = self.kmer_length
        # kmer_length = max(kmer_length, 5)
        start_idx = int(np.random.rand() * (self.minlen - kmer_length))
        # make a kmer seed
        kmer_seed = seed_sequence[start_idx : start_idx + kmer_length]
        random_seq = generate_string_sequence(self.minlen)
        start_idx = int(np.random.rand() * (self.minlen - kmer_length))
        seeded_seq = random_seq[:start_idx] + kmer_seed + random_seq[start_idx + kmer_length :]
        # take the Mel
        self.idx_cnt += 1
        # just do pairs for now.
        return (
            torch.log(
                1
                + self.mel(
                    torch.as_tensor(
                        [utils.amino_char_to_index[c] for c in sanitize_sequence(seed_sequence)]
                    ).float()
                )
            ),
            torch.log(
                1
                + self.mel(
                    torch.as_tensor(
                        [utils.amino_char_to_index[c] for c in sanitize_sequence(seeded_seq)]
                    ).float()
                )
            ),
            self.idx_cnt,
        )


class AAIndexDataset(DataModule):
    def __init__(self, fa_file, minlen, training):
        _, seqs = utils.fasta_from_file(fa_file)
        self.seqs = [s for s in seqs if len(s) > minlen]
        self.minlen = minlen
        self.training = training

        self.sub_dists = utils.create_substitution_distribution(62)
        self.sub_probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        with open("src/resources/indices.txt") as f:
            data = f.read()
        split = data.split("//")
        indices = [s[s.find("I") :].replace("\n", "").split() for s in split]

        aas = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]

        self.indices = []

        for index in indices:
            if len(index) <= 20:
                logger.debug("Skipping index.")
                continue

            mapping = AAIndexFFT()
            broke = False
            for i, aa in enumerate(aas):
                try:
                    mapping[aa] = float(index[-(20 - i)])
                except ValueError:
                    broke = True
                    break
            if not broke:
                self.indices.append(mapping)

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
            return (
                torch.stack(data),
                torch.as_tensor(labels),
            )

        return pad_view_batches

    def __len__(self) -> int:
        if self.training:
            return len(self.seqs)
        else:
            return 10000

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        # subsample sequence;
        if len(seq) != self.minlen:
            start_idx = np.random.randint(0, len(seq) - self.minlen)
        else:
            start_idx = 0
        seq = seq[start_idx : start_idx + self.minlen]

        sequence = sanitize_sequence(seq)

        sequence = torch.as_tensor([utils.amino_char_to_index[c] for c in sequence])

        n_subs = int(len(sequence) * self.sub_probs[np.random.randint(0, len(self.sub_probs))])

        s2 = utils.mutate_sequence(
            sequence=sequence, substitutions=n_subs, sub_distributions=self.sub_dists,
        )
        # this creates a fuzzy tensor.
        # map back to character space;
        s1 = torch.log(1 + self.mel(sequence).float())
        s2 = torch.log(1 + self.mel(s2).float())

        return s1, s2, idx % len(self.seqs)


class UniProtSpectDataset(SwissProtGenerator):
    def __init__(self, *args, **kwargs):
        super(UniProtSpectDataset, self).__init__(*args, **kwargs)
        self.mel = MelSpectrogram(win_length=32, hop_length=16, n_mels=50)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        # subsample sequence;
        if len(seq) != self.minlen:
            start_idx = np.random.randint(0, len(seq) - self.minlen)
        else:
            start_idx = 0
        seq = seq[start_idx : start_idx + self.minlen]

        sequence = sanitize_sequence(seq)

        sequence = torch.as_tensor([utils.amino_char_to_index[c] for c in sequence])

        n_subs = int(len(sequence) * self.sub_probs[np.random.randint(0, len(self.sub_probs))])

        s2 = utils.mutate_sequence(
            sequence=sequence, substitutions=n_subs, sub_distributions=self.sub_dists,
        )

        s1 = self.mel(sequence.float())
        s1 = s1 / torch.max(s1)
        s2 = self.mel(s2.float())
        s2 = s2 / torch.max(s2)
        s1[torch.isnan(s1)] = 0
        s2[torch.isnan(s2)] = 0

        return s1, s2, idx % len(self.seqs)

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
            return (
                torch.stack(data),
                torch.as_tensor(labels),
            )

        return pad_view_batches
