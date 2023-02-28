
# pylint: disable=no-member
import torch
import src.utils as utils
from src.datasets import DataModule
from src.utils.gen_utils import generate_string_sequence
from src.datasets.datasets import sanitize_sequence
import pdb

# class AlignmentGenerator(DataModule):

#     def __init__(self, ali_path, seq_len, training=True):
#         f = open(ali_path, 'r')
#         self.alignment_file_paths = f.readlines()
#         f.close()
#         print(f"Found {len(self.alignment_file_paths)} alignment files")

#         self.training = training
#         self.seq_len = seq_len
#         print(f'Sequence length: {self.seq_len}')
#         self.mx = 0

#     def collate_fn(self):
#         return None

#     def __len__(self):
#         return len(self.alignment_file_paths)

#     def parse_alignment(self, alignment_file: str):
#         """Returns the aligned query and target
#         sequences from an alignment file"""
#         f = open(alignment_file, "r")
#         lines = f.readlines()
#         assert len(lines) == 3
#         seq1 = lines[1].strip("\n")
#         seq2 = lines[2].strip("\n")
#         f.close()
#         return seq1.upper(), seq2.upper()

#     def parse_indels(self, sequence1: str, sequence2: str):
#         pass


#     def pad_stop_codons(self, sequence, indices, value):
#         sequence_length = self.seq_len 

#         if len(sequence) > sequence_length:
#             sequence = sequence[-sequence_length:]
#             indices = indices[-sequence_length:]
# # #             seq2 = addition + seq2
# # #             seq1 = addition + seq1
# #             for i in range(5):
# #                 sequence += '*'
# #                 indices.append(value)
#         elif len(sequence) < sequence_length:
#             seq_chop = len(sequence) - self.seq_len
#             addition = generate_string_sequence(-seq_chop)
#            # addition_length = sequence_length - len(sequence)
#             sequence = addition + sequence
#             indices += [value] * -seq_chop
#         return sequence, indices

#     def __getitem__(self, idx):

#         alignment_path = self.alignment_file_paths[idx].strip('\n')

#         seq1, seq2 = self.parse_alignment(alignment_path)

#         seq1 = sanitize_sequence(seq1)
#         seq2 = sanitize_sequence(seq2)

#         seq_a_aligned_labels = list(range(self.mx, self.mx + len(seq1)))
#         seq_b_aligned_labels = list(range(self.mx, self.mx + len(seq2)))
#         # just remove elements from the labels above
#         # that are gap characters in either sequence.
#         # then labels that are the same will be aligned characters
#         seq_a_to_keep = []
#         for j, amino in enumerate(seq1):
#             if amino not in ("-", "."):
#                 seq_a_to_keep.append(j)

#         seq_a_aligned_labels = [seq_a_aligned_labels[s] for s in seq_a_to_keep]
#         seq_b_to_keep = []
#         for j, amino in enumerate(seq2):
#             if amino not in ("-", "."):
#                 seq_b_to_keep.append(j)

#         seq_b_aligned_labels = [seq_b_aligned_labels[s] for s in seq_b_to_keep]
#         seq1 = seq1.replace(".", "").replace("-", "")
#         seq2 = seq2.replace(".", "").replace("-", "")

#         seq1_chop = len(seq1) - self.seq_len
#         seq2_chop = len(seq2) - self.seq_len

#         if seq1_chop > 0:
#             seq1 = seq1[seq1_chop // 2 : -seq1_chop // 2]
#             seq_a_aligned_labels = seq_a_aligned_labels[
#                 seq1_chop // 2 : -seq1_chop // 2
#             ]
#             # seq1 = seq1[-seq1_chop:]
#             # seq_a_aligned_labels = seq_a_aligned_labels[-seq1_chop:]
#         elif seq1_chop == 0:
#             pass
#         else:
#             # add characters to the front
#             addition = generate_string_sequence(-seq1_chop)
#             # now add bullshit to the labels at the beginning
#             mx = max(max(seq_a_aligned_labels), max(seq_b_aligned_labels)) + 1
#             seq1 = addition + seq1
#             for _ in range(len(addition)):
#                 seq_a_aligned_labels.insert(0, mx)
#                 mx += 1

#         if seq2_chop > 0:
#             seq2 = seq2[seq2_chop // 2 : -seq2_chop // 2]
#             seq_b_aligned_labels = seq_b_aligned_labels[
#                 seq2_chop // 2 : -seq2_chop // 2
#             ]
#             # seq2 = seq2[-seq2_chop:]
#             # seq_b_aligned_labels = seq_b_aligned_labels[-seq2_chop:]
#         elif seq2_chop == 0:
#             pass
#         else:
#             # add characters to the front
#             addition = generate_string_sequence(-seq2_chop)
#             # now add bullshit to the labels at the beginning
#             mx = max(max(seq_a_aligned_labels), max(seq_b_aligned_labels)) + 1
#             seq2 = addition + seq2
#             for _ in range(len(addition)):
#                 seq_b_aligned_labels.insert(0, mx)
#                 mx += 1
#         # brutally chop off the ends?
#         self.mx = max(max(seq_a_aligned_labels), max(seq_b_aligned_labels)) + 1
#         return (
#             utils.encode_string_sequence(seq1),
#             torch.as_tensor(seq_a_aligned_labels),
#             utils.encode_string_sequence(seq2),
#             torch.as_tensor(seq_b_aligned_labels),
#         )









        # seq1, seq1_indices = self.pad_stop_codons(seq1, seq1_indices, float('nan'))
        # seq2, seq2_indices = self.pad_stop_codons(seq2, seq2_indices, float('nan'))
        # assert len(seq1) == len(seq1_indices) == len(seq2_indices) == len(seq2)


        # seq1_m = utils.encode_string_sequence(seq1)
        # seq2_m = utils.encode_string_sequence(seq2)
        # return seq1_m, seq1_indices, seq2_m, seq2_indices



class AlignmentGeneratorWithIndels(DataModule):

    def __init__(self, ali_path, seq_len, training=True):
        f = open(ali_path, 'r')
        self.alignment_file_paths = f.readlines()
        f.close()
        print(f"Found {len(self.alignment_file_paths)} alignment files")

        self.training = training
        self.seq_len = seq_len
        print(f'Sequence length: {self.seq_len}')

    def collate_fn(self):
        return None

    def __len__(self):
        return len(self.alignment_file_paths)

    def parse_alignment(self, alignment_file: str):
        """Returns the aligned query and target
        sequences from an alignment file"""
        f = open(alignment_file, "r")
        lines = f.readlines()
        assert len(lines) == 3
        seq1 = lines[1].strip("\n")
        seq2 = lines[2].strip("\n")
        f.close()
        return seq1.upper(), seq2.upper()

    def parse_indels(self, sequence1: str, sequence2: str):
        seq1 = sanitize_sequence(sequence1)
        seq2 = sanitize_sequence(sequence2)

        length = len(seq1)

        #TODO: keep trakc of indices 
        seq1_dots_and_dashes = [i for i in range(length) if seq1[i] == "." or seq1[i] == "-"]
        seq2_dots_and_dashes = [i for i in range(length) if seq2[i] == "." or seq2[i] == "-"]

        seq1_indices = []
        seq2_indices = []
        for i in range(length):
            if i in seq2_dots_and_dashes:
                continue
            elif i in seq1_dots_and_dashes:
                seq2_indices.append(float('nan'))
            else:
                seq2_indices.append(i)
        for i in range(length):
            if i in seq1_dots_and_dashes:
                continue
            elif i in seq2_dots_and_dashes:
                seq1_indices.append(float('nan'))
            else:
                seq1_indices.append(i)
        seq1_without_gaps = seq1.replace("-","").replace(".","")

        seq2_without_gaps = seq2.replace("-","").replace(".","")

        pdb.set_trace()

        return seq1_without_gaps, seq1_indices, seq2_without_gaps, seq2_indices

    def pad_stop_codons(self, sequence, indices, value):
        sequence_length = self.seq_len 

        if len(sequence) > sequence_length:
            sequence = sequence[-sequence_length:]
            indices = indices[-sequence_length:]
# #             seq2 = addition + seq2
# #             seq1 = addition + seq1
#             for i in range(5):
#                 sequence += '*'
#                 indices.append(value)
        elif len(sequence) < sequence_length:
            seq_chop = len(sequence) - self.seq_len
            addition = generate_string_sequence(-seq_chop)
           # addition_length = sequence_length - len(sequence)
            sequence = addition + sequence
            indices += [value] * -seq_chop
        return sequence, indices

    def __getitem__(self, idx):

        alignment_path = self.alignment_file_paths[idx].strip('\n')

        seq1, seq2 = self.parse_alignment(alignment_path)

        seq1, seq1_indices, seq2, seq2_indices = self.parse_indels(seq1, seq2)
        seq1, seq1_indices = self.pad_stop_codons(seq1, seq1_indices, float('nan'))
        seq2, seq2_indices = self.pad_stop_codons(seq2, seq2_indices, float('nan'))
        assert len(seq1) == len(seq1_indices) == len(seq2_indices) == len(seq2)


        seq1_m = utils.encode_string_sequence(seq1)
        seq2_m = utils.encode_string_sequence(seq2)
        return seq1_m.float(), torch.as_tensor(seq1_indices).float(), seq2_m.float(), torch.as_tensor(seq2_indices).float()

