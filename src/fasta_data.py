from Bio import SeqIO
from collections import defaultdict
import torch

class FASTAData:
    """
    Read a FASTA file and bucket its sequences by length.

    Each sequence is tokenized and stored in per-length buckets
    so that downstream models can operate on batches of
    equal-length sequences.

    Parameters
    ----------
    file_path : str
        Path to the FASTA file to parse.
    min_seq_length : int, default 128
        Sequences shorter than this threshold are ignored.
    max_seq_length : int, default 99_999_999
        Sequences longer than this threshold are ignored.

    Attributes
    ----------
    tokens_by_length : dict[int, torch.Tensor]
        Maps *L* → 2-D tensor of integer-encoded residues with shape
        ``(n_sequences_of_length_L, L)``.
    masks_by_length : dict[int, torch.Tensor]
        Maps *L* → Boolean mask array with the same shape as
        ``tokens_by_length[L]``; ``True`` marks real residues and
        ``False`` marks residues that should be ignored during search.
    seqids_by_length : dict[int, torch.Tensor]
        Maps *L* → 1-D array of internal sequence IDs that index
        into `seqid_to_name`.
    seqid_to_name : list[str]
        Lookup table that converts an internal sequence ID into the
        original FASTA record header.
    """
    def __init__(self, file_path: str, min_seq_length: int=128, max_seq_length: int=99999999):
        self.tokens_by_length = dict()
        self.masks_by_length = dict()
        self.seqids_by_length = dict()
        self.seqid_to_name = list()

    """
        Read a FASTA file and bucket its sequences by length.

        Parameters
        ----------
        file_path : str
            Path to the FASTA file to parse.
        min_seq_length : int, default 128
            Sequences shorter than this threshold are ignored.
        max_seq_length : int, default 99_999_999
            Sequences longer than this threshold are ignored.
    """
    def read_fasta(self, fasta_path: str, min_seq_length: int=128, max_seq_length: int=99999999):
        amino_acids = 'XARNDCEQGHILKMFPSTWYVBJZ0'
        alphabet = {aa: i for i, aa in enumerate(amino_acids)}
        alphabet['U'] = 0
        alphabet['B'] = 0
        alphabet['J'] = 0
        alphabet['Z'] = 0
        alphabet['0'] = 0

        # We first load the data into memory
        with open(fasta_path) as handle:
            seq_names_by_length: dict[int, list] = defaultdict(list)
            seq_strings_by_length: dict[int, list]  = defaultdict(list)

            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                name = record.id
                length = len(seq)
                if length < 32:
                    continue
                seq_names_by_length[length].append(name)
                seq_strings_by_length[length].append(seq)

        # Then we tokenize the data
        for length in seq_strings_by_length:
            num_seqs = len(seq_strings_by_length[length])
            masks = torch.ones(num_seqs, length, dtype=torch.bool)
            seqs = torch.zeros(num_seqs, length, dtype=torch.int)

            for i, s in enumerate(seq_strings_by_length[length]):
                seqs[i] = torch.tensor(list(s.upper().encode('ascii')), dtype=torch.int)
                masks[i] = torch.tensor([True if c.isupper() else False for c in s])

            for c in amino_acids:
                seqs[seqs == ord(c)] = alphabet[c]
            seqs[seqs > len(alphabet)] = 0

            self.masks_by_length[length] = masks
            self.tokens_by_length[length] = seqs
            self.seqids_by_length[length] = torch.arange(len(self.seqid_to_name),
                                                         len(self.seqid_to_name)+len(self.tokens_by_length[length]))
            self.seqid_to_name.extend(seq_names_by_length[length])