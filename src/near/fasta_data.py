from Bio import SeqIO
from collections import defaultdict
import torch
import numpy as np

class FASTAData:
    def __init__(self, file_path: str, min_seq_length: int=128, max_seq_length: int=99999999):
        self.tokens_by_length = {}
        self.masks_by_length = {}
        self.seqids_by_length = {}
        self.tokenids_by_length = {}
        self.seqid_to_name = []
        self.seqid_to_length = None
        self.read_fasta(file_path, min_seq_length, max_seq_length)

    def read_fasta(self, fasta_path: str, min_seq_length: int=128, max_seq_length: int=99999999):
        amino_acids = 'XARNDCEQGHILKMFPSTWYV'
        alphabet = np.zeros(256, dtype=np.int64)
        alphabet[[ord(c) for c in amino_acids]] = np.arange(len(amino_acids))

        seq_names_by_length = defaultdict(list)
        seq_strings_by_length = defaultdict(list)

        total_sequences = 0

        with open(fasta_path, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                length = len(seq)
                if length < min_seq_length or length > max_seq_length:
                    continue
                seq_names_by_length[length].append(record.id)
                seq_strings_by_length[length].append(seq)
                total_sequences += 1

        self.seqid_to_length = np.zeros(total_sequences, dtype=np.uint64)

        seq_id_counter = 0

        for length, sequences in seq_strings_by_length.items():
            num_seqs = len(sequences)

            seq_array = np.zeros((num_seqs, length), dtype=np.uint8)
            masks = np.ones((num_seqs, length), dtype=bool)

            for i, seq in enumerate(sequences):
                seq_bytes = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
                seq_array[i, :] = seq_bytes
                masks[i, :] = np.array([c.isupper() for c in seq])

            tokens = alphabet[seq_array]
            masks &= tokens > 0

            self.tokens_by_length[length] = torch.from_numpy(tokens)
            self.masks_by_length[length] = torch.from_numpy(masks)

            seq_ids = np.arange(seq_id_counter, seq_id_counter + num_seqs, dtype=np.uint64)
            seq_id_counter += num_seqs
            self.seqids_by_length[length] = seq_ids

            self.seqid_to_length[seq_ids] = masks.sum(axis=1)

            token_ids = np.arange(length, dtype=np.uint64)
            end_cutoff = length - 64
            middle_emb = (token_ids >= 63) & (token_ids <= end_cutoff)
            token_ids[token_ids > end_cutoff] = (127 - (length - token_ids[token_ids > end_cutoff]))
            token_ids[middle_emb] = 64

            token_pos = (np.arange(length, dtype=np.uint64) << 7)
            token_ids = (seq_ids[:, None] << 32) | token_ids[None, :] | token_pos[None, :]

            self.tokenids_by_length[length] = token_ids

            self.seqid_to_name.extend(seq_names_by_length[length])
