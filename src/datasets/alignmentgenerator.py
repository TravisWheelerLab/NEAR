"""Alignment generator class for using HMMER alignments parsed from the 
HMMER stdout as data, with indels that are masked by indicess"""
import torch
from src import utils
from src.datasets import DataModule
from src.utils.gen_utils import generate_string_sequence


class AlignmentGenerator(DataModule):
    """Alignment generator class without indels"""

    def collate_fn(self):
        """What is returned when fetching batch"""

        def pad(batch):
            """
            Pad batches with views as a dim.
            Input: [n_viewsx...]
            :param batch: list of np.ndarrays encoding protein sequences/logos
            :type batch: List[np.ndarray]
            :return: torch.tensor
            :rtype: torch.tensor
            """

            seq1 = [b[0] for b in batch]
            seq2 = [b[1] for b in batch]
            data = seq1 + seq2
            # seq1_raw = [b[2] for b in batch]
            # seq2_raw = [b[3] for b in batch]
            return torch.stack(data)  # seq1_raw, seq2_raw

        return pad

    def __init__(self, ali_path, seq_len, training=True):
        """Arguments:
        ali_path: path to alignments
        seq_len: fixed sequence length"""

        with open(ali_path, "r") as file:
            self.alignment_file_paths = file.readlines()
        print(f"Found {len(self.alignment_file_paths)} alignment files")

        self.training = training
        self.seq_len = seq_len

    def __len__(self):
        """Number of alignment files"""
        return len(self.alignment_file_paths)

    def parse_alignment(self, alignment_file: str):
        """Returns the aligned query and target
        sequences from an alignment file"""
        with open(alignment_file, "r") as file:
            lines = file.readlines()
            seq1 = lines[1].strip("\n")
            seq2 = lines[2].strip("\n")
        return seq1.upper(), seq2.upper()

    def __getitem__(self, idx):
        """Get ungapped alignments"""

        alignment_path = self.alignment_file_paths[idx].strip("\n")

        seq1, seq2 = self.parse_alignment(alignment_path)
        assert len(seq1) == len(seq2)

        seq1_dots_and_dashes = [
            i for i in range(len(seq1)) if seq1[i] == "." or seq1[i] == "-"
        ]
        seq2_dots_and_dashes = [
            i for i in range(len(seq2)) if seq2[i] == "." or seq2[i] == "-"
        ]

        clean_seq1 = ""
        clean_seq2 = ""
        for i in range(len(seq1)):
            if i not in seq1_dots_and_dashes and i not in seq2_dots_and_dashes:
                clean_seq1 += seq1[i]
                clean_seq2 += seq2[i]

        seq1 = clean_seq1
        seq2 = clean_seq2

        if len(seq1) > self.seq_len:
            seq1 = seq1[-self.seq_len :]
            seq2 = seq2[-self.seq_len :]
        elif len(seq1) < self.seq_len:
            seq_chop = len(seq1) - self.seq_len
            addition1 = generate_string_sequence(-seq_chop)
            addition2 = generate_string_sequence(-seq_chop)

            seq2 = addition1 + seq2
            seq1 = addition2 + seq1

        return utils.encode_string_sequence(
            seq1
        ), utils.encode_string_sequence(seq2)


class AlignmentGeneratorWithIndels(DataModule):
    """Alignment generator with insertions and deletions"""

    def __init__(self, ali_path, seq_len, training=True):
        """Arguments:
        ali_path: path to alignments
        seq_len: fixed sequence length"""

        with open(ali_path, "r") as file:
            self.alignment_file_paths = file.readlines()
        print(f"Found {len(self.alignment_file_paths)} alignment files")

        self.training = training
        self.seq_len = seq_len

    def __len__(self):
        """Number of alignment files"""
        return len(self.alignment_file_paths)

    def collate_fn(self):
        """What is returned when getting a batch
        if None, defaults to the return of __getitem__()"""
        return None

    def parse_alignment(self, alignment_file: str):
        """Returns the aligned query and target
        sequences from an alignment file"""
        with open(alignment_file, "r") as file:
            lines = file.readlines()
            seq1 = lines[1].strip("\n")
            seq2 = lines[2].strip("\n")
            seq1_full = lines[3].strip("\n")
            seq2_full = lines[4].strip("\n")
        return seq1.upper(), seq2.upper(), seq1_full.upper(), seq2_full.upper()

    def parse_indels(self, seq1: str, seq2: str):
        """
        Removes dots and dashes
        and keeps track of where the indels are so that
        we can then apply a mask on the loss function
        """

        length = len(seq1)

        # indices of dots and dashes
        seq1_dots_and_dashes = [
            i for i in range(length) if seq1[i] == "." or seq1[i] == "-"
        ]
        seq2_dots_and_dashes = [
            i for i in range(length) if seq2[i] == "." or seq2[i] == "-"
        ]

        # keep track of indices of aligned aminos
        # put 'nan' in where amino has no aligned amino in corresponding sequence
        seq1_indices = []
        seq2_indices = []
        for i in range(length):
            if i in seq2_dots_and_dashes:
                continue
            if i in seq1_dots_and_dashes:
                seq2_indices.append(float("nan"))
            else:
                seq2_indices.append(i)
        for i in range(length):
            if i in seq1_dots_and_dashes:
                continue
            if i in seq2_dots_and_dashes:
                seq1_indices.append(float("nan"))
            else:
                seq1_indices.append(i)

        # remove dots and dashes
        seq1_without_gaps = seq1.replace("-", "").replace(".", "")
        seq2_without_gaps = seq2.replace("-", "").replace(".", "")

        return seq1_without_gaps, seq1_indices, seq2_without_gaps, seq2_indices

    def pad_sequence(
        self,
        sequence: str,
        indices: list,
        subseq_index: int,
        full_seq: str,
        left_padding: int,
        right_padding: int,
        value=float("nan"),
    ):
        """Pads the sequence so that it has length self.seq_len
        We use the full sequence to pad around the subequence,
        this helps maintain context in the CNN"""
        addition_left = full_seq[subseq_index - left_padding : subseq_index]
        addition_right = full_seq[
            subseq_index
            + len(sequence) : subseq_index
            + len(sequence)
            + right_padding
        ]
        sequence = addition_left + sequence + addition_right
        assert (
            full_seq.find(sequence) != -1
        ), f"new sequence {sequence} is not part of full sequence + \n + {full_seq}"
        indices = [value] * left_padding + indices + [value] * right_padding

        return sequence, indices

    def fix_sequence_length(
        self, sequence: str, indices: list, full_seq: str, value=float("nan")
    ):
        """Currently we are padding sequences that are less
        than sequence length with randomly generated string sequence
        according to a background distribution.
        However we put in 'nan' in the indices so these aminos will not
        be aligned in the loss function
        if the sequence is longer than the sequence length then
        we can just chop them"""

        sequence_length = self.seq_len
        if len(sequence) >= sequence_length:
            sequence = sequence[:sequence_length]
            indices = indices[:sequence_length]
            return sequence, indices

        if len(sequence) < sequence_length:
            seq_chop = self.seq_len - len(sequence)
            subseq_index = full_seq.find(sequence)
            half = seq_chop // 2
            if (
                len(full_seq) >= self.seq_len
            ):  # we can get all our padding from the full sequence
                if (seq_chop) % 2 == 0:  # chop sequence is even
                    addition_left_amt = half
                    addition_right_amt = half
                else:
                    addition_left_amt = half
                    addition_right_amt = half + 1

                if (
                    subseq_index >= addition_left_amt
                    and len(full_seq) - (subseq_index + len(sequence))
                    >= addition_right_amt
                ):
                    sequence, indices = self.pad_sequence(
                        sequence,
                        indices,
                        subseq_index,
                        full_seq,
                        addition_left_amt,
                        addition_right_amt,
                    )
                    return sequence, indices

                if subseq_index < addition_left_amt:
                    addition_right_amt = addition_right_amt + (
                        addition_left_amt - subseq_index
                    )
                    addition_left_amt = subseq_index

                    sequence, indices = self.pad_sequence(
                        sequence,
                        indices,
                        subseq_index,
                        full_seq,
                        addition_left_amt,
                        addition_right_amt,
                    )
                    return sequence, indices

                if len(full_seq) - (subseq_index + len(sequence)) < half:
                    addition_right_amt = len(full_seq) - (
                        subseq_index + len(sequence)
                    )

                    addition_left_amt = (
                        self.seq_len - addition_right_amt - len(sequence)
                    )
                    sequence, indices = self.pad_sequence(
                        sequence,
                        indices,
                        subseq_index,
                        full_seq,
                        addition_left_amt,
                        addition_right_amt,
                    )
                    return sequence, indices
                addition_left_amt = half + 1
                addition_right_amt = half
                sequence, indices = self.pad_sequence(
                    sequence,
                    indices,
                    subseq_index,
                    full_seq,
                    addition_left_amt,
                    addition_right_amt,
                )
                return sequence, indices
            addition_left_amt = subseq_index
            addition_right_amt = len(full_seq) - (subseq_index + len(sequence))
            sequence, indices = self.pad_sequence(
                sequence,
                indices,
                subseq_index,
                full_seq,
                addition_left_amt,
                addition_right_amt,
            )
            seq_chop = self.seq_len - len(sequence)

            addition = generate_string_sequence(seq_chop)
            sequence = sequence + addition
            indices = indices + [value] * seq_chop

            return sequence, indices

    def __getitem__(self, idx: int):
        """Gets the sequences as indices for a batch"""
        alignment_path = self.alignment_file_paths[idx].strip("\n")

        seq1_raw, seq2_raw, seq1_full, seq2_full = self.parse_alignment(
            alignment_path
        )

        seq1, seq1_indices, seq2, seq2_indices = self.parse_indels(
            seq1_raw, seq2_raw
        )
        seq1, seq1_indices = self.fix_sequence_length(
            seq1, seq1_indices, seq1_full
        )
        seq2, seq2_indices = self.fix_sequence_length(
            seq2, seq2_indices, seq2_full
        )

        assert (
            len(seq1) == len(seq1_indices) == len(seq2_indices) == len(seq2)
        ), print(
            f"Not all the same length! {idx} {seq1} {seq1_full} {seq2} {seq2_full}"
        )

        seq1_m = utils.encode_string_sequence(seq1)
        seq2_m = utils.encode_string_sequence(seq2)
        return (
            seq1_m.float(),
            torch.as_tensor(seq1_indices).float(),
            seq2_m.float(),
            torch.as_tensor(seq2_indices).float(),
        )  # , seq1, seq2, seq1_raw, seq2_raw
