# Standard library imports
import os
import random
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Tuple
from pathlib import Path


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F

# Biopython imports
from Bio import AlignIO, SeqIO

        
class AlignmentDataset(Dataset):
    def __init__(self, alignment_dir: str,
                 query_dir: str, 
                 target_dir:str, 
                 seq_length: int, 
                 max_offset: int,
                 min_aligned_tokens: int=12,
                 use_random_seq_length: bool = False,
                 random_seq_min_length: int=29,
                 random_mask_rate = 0.0,
                 softmask=True):

        self.alignment_dir = alignment_dir
        self.query_dir = query_dir
        self.target_dir = target_dir

        self.seq_length = seq_length
        self.max_offset = max_offset
        self.offset_ratio = max_offset / seq_length

        self.min_aligned_tokens = min_aligned_tokens

        self.use_random_seq_length = use_random_seq_length
        self.random_seq_min_length = random_seq_min_length

        self.random_seq_lengths = np.arange(random_seq_min_length, seq_length+1)
        self.random_seq_lengths = self.random_seq_lengths[-1] - self.random_seq_lengths
        self.random_seq_lengths += random_seq_min_length

        self.random_seq_dist = np.exp(np.linspace(0, 1, len(self.random_seq_lengths)))
        self.random_seq_dist = self.random_seq_dist / np.sum(self.random_seq_dist)
        self.random_seq_dist[0] += 1.0
        self.random_seq_dist = self.random_seq_dist / 2.0

        self.random_mask_rate = random_mask_rate

        self.amino_acids = 'XARNDCEQGHILKMFPSTWYVBJZ0'
        self.normal_aa = [c for c in 'ARNDCEQGHILKMFPSTWYV']
        self.amino_to_index = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.amino_to_index['U'] = 0
        self.amino_to_index['B'] = 0
        self.amino_to_index['J'] = 0
        self.amino_to_index['Z'] = 0
        self.amino_to_index['0'] = 0
        self.softmask=softmask

        self._create_file_list()
        self._create_alignment_indices()

    def _create_file_list(self):
        """
        Create a list of file tuples containing alignment, query, and target files.
        Each tuple contains paths to corresponding alignments (stockholm format), query (fasta format), and target (fasta format) files.
        """
        files = []
    
        for alignment_file in os.listdir(self.alignment_dir):
            # Construct full file paths
            alignment_path = os.path.join(self.alignment_dir, alignment_file)
            if os.path.isfile(alignment_path):
                query_path = os.path.join(self.query_dir, alignment_file)
                target_path = os.path.join(self.target_dir, alignment_file)

                # Add the file tuple to the list
                files.append((alignment_path, query_path, target_path))
        
        self.file_list = files

    def _create_alignment_indices(self):
        alignments_per_file = torch.zeros(len(self.file_list)).int()
        i = 0
        for alignment_file, _, _ in self.file_list:
            count = 0
            with open(alignment_file, 'r') as file:
                for line in file:
                    if len(line) < 5:
                        continue
                    if line.startswith('  =='):
                        count += 1
            alignments_per_file[i] = count
            i = i + 1
        self.alignments_per_file = alignments_per_file
        idx_to_file = torch.zeros(alignments_per_file.sum().item()).int()
        previous_count = 0
        for i, count in enumerate(self.alignments_per_file):
            count = count.item()
            idx_to_file[previous_count:previous_count + count] = i
            previous_count += count
        self.idx_to_file = idx_to_file

    def __len__(self) -> int:
        """Return the number of file tuples in the dataset."""
        return len(self.idx_to_file)

    def _load_seq_file(self, file_path: str) -> Dict[str, str]:
        """
        Load sequences from a FASTA file.
        
        Args:
            file_path (str): Path to the FASTA file.
        
        Returns:
            Dict[str, str]: A dictionary mapping sequence IDs to their sequences.
        """
        sequences =  {k: str(v.seq) for k, v in SeqIO.to_dict(SeqIO.parse(file_path, "fasta")).items()}
        return sequences

    def read_alignment_line_sto(self, line: str) -> Tuple[str, int, int, str]:
        line = line.split('/')
        seq_name = line[0]
        line = line[1].split('-', maxsplit=1)
        start = int(line[0]) - 1
        line = line[1].split(' ', maxsplit=1)
        end = int(line[0])
        seq = line[1].lstrip(' ').rstrip(' ')
        
        return (seq_name, start, end, seq)
        
    def read_alignment_block_sto(self, alignment_block: str) -> Tuple[str, str, List[Tuple[int, int, str]]]:
        # Parse block
        # Search for query ID
        # Search for alignment lines
        
        query_name = ""
        alignments = defaultdict(list)
        
        looking_for_query = True
        for line in alignment_block.split('\n'):
            line = line.rstrip()
            if len(line) < 2:
                continue
            if looking_for_query:
                if line.startswith('#=GF ID'):
                    query_name = line[8:]
                    looking_for_query = False
            else:
                if line[0] != '#':
                    target_name, start, end, seq = self.read_alignment_line(line)
                    alignments[target_name].append((start, end, seq))

        if looking_for_query:
            print(alignment_block)
            raise ValueError("Query not found in alignment block")
        return (query_name, dict(alignments))

    def _load_alignment_file_sto(self, file_path: str) -> Dict[str, Dict[str, List[Tuple[int, int, str]]]]:
        """
        Load alignments from a Stockholm format file.
        
        Args:
            file_path (str): Path to the Stockholm format file.
        
        Returns:
            Dict[str, Dict[str, List[Tuple[int, int, str]]]]: A nested dictionary structure containing alignment information.
        """
        with open(file_path, 'r') as content_file:
            file_contents = content_file.read()
        
        # Split the file contents into individual alignments
        file_contents = file_contents.split('//\n')
        alignment_dict = defaultdict(lambda: defaultdict(list))
        
        for alignment_block in file_contents:
            if len(alignment_block) < 50:
                continue
            
            query_name, block_alignments = self.read_alignment_block(alignment_block)
            alignment_dict[query_name] = block_alignments
            
        return dict(alignment_dict)
    
    def _load_alignment_file(self, file_path: str) -> List[Tuple[str, int, str, str, int, str]]:
        """
        Load alignments from a phmmer output file (expects --notextw).
        
        Args:
            file_path (str): Path to the HMMER alignment file.
        
        Returns:
            Dict[str, Dict[str, List[Tuple[int, int, str]]]]: A nested dictionary structure containing alignment information.
        """

        # 0 waiting for alignment ('==')
        # 1 query seq line
        # 2 ignore line
        # 3 target seq line
        parser_state = 0
        query_name = None
        target_name = None
        query_start = 0
        target_start = 0
        query_seq = None
        target_seq = None

        alignments = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.rstrip().lstrip(' ')
                
                if parser_state == 0:
                    if line.startswith('=='):
                        parser_state = 1

                elif parser_state == 1:
                    line = line.split(' ', maxsplit=1)
                    query_name = line[0]
                    line = line[1].lstrip(' ')
                    line = line.split(' ', maxsplit=1)
                    query_start = int(line[0])
                    line = line[1].lstrip(' ')
                    line = line.split(' ', maxsplit=1)
                    query_seq = line[0].upper()

                    parser_state = 2

                elif parser_state == 2:
                    parser_state = 3

                elif parser_state == 3:
                    line = line.split(' ', maxsplit=1)
                    target_name = line[0]
                    line = line[1].lstrip(' ')
                    line = line.split(' ', maxsplit=1)
                    target_start = int(line[0])
                    line = line[1].lstrip(' ')
                    line = line.split(' ', maxsplit=1)
                    target_seq = line[0].upper()
                    parser_state = 0
                    alignments.append((query_name, query_start, query_seq, target_name, target_start, target_seq))
        return alignments


                

    def _create_sequence_tensors(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert an amino acid sequence to tensors.

        Args:
            seq (str): Input amino acid sequence.

        Returns:
            tuple: A tuple containing:
                - seq (torch.Tensor): Tensor of amino acid indices.
                - mask (torch.Tensor): Mask tensor indicating lowercase (modified) amino acids.
        """
        # Define amino acid vocabulary (0 is reserved for padding/null)
        
        
        # Create mask tensor (True for lowercase/modified amino acids)

        if len(seq) < self.seq_length:
            new_characters = ''.join(np.random.choice(self.normal_aa, size=self.seq_length-len(seq)))
            seq = seq + new_characters

        if self.softmask:
            mask = torch.tensor([c.isupper() for c in seq])
        else:
            mask = torch.ones(len(seq), dtype=torch.bool)
        m = ''.join([str(int(c.isupper())) for c in seq])
        

        # Convert sequence to tensor of indices
        seq = torch.tensor([self.amino_to_index[c] for c in seq.upper()], dtype=torch.int64)
        return seq, mask
    
    def _create_training_data(self, alignment: Tuple[int, int, str],
                                  query_seq: torch.Tensor, query_mask: torch.Tensor,
                                  target_seq: torch.Tensor, target_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create training matrices from alignment and sequence data.

        Args:
            alignment (Tuple[int, int, str]): Alignment information (start, end, sequence).
            query_seq (torch.Tensor): Query sequence tensor.
            query_mask (torch.Tensor): Query sequence mask tensor.
            target_seq (torch.Tensor): Target sequence tensor.
            target_mask (torch.Tensor): Target sequence mask tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Target matrix and mask matrix.
        """
        query_name, query_start, a_query = alignment[0], alignment[1] - 1, alignment[2]
        target_name, target_start, a_target = alignment[3], alignment[4] - 1, alignment[5]
        #target_seq = target_seq[alignment_start:]
        
        # Initialize matrices
        target_matrix = torch.zeros((self.seq_length, self.seq_length))
        mask_matrix = torch.zeros((self.seq_length, self.seq_length), dtype=torch.bool)
        
        # Create position tensors
        query_pos = torch.zeros(len(query_seq)) - 1.0
        target_pos = torch.zeros(len(target_seq)) - 2.0

        # Parse alignment
        q_i = query_start
        t_i = target_start
        for i, (q, t) in enumerate(zip(a_query, a_target)):
            #if q_i < len(query_seq) and t_i < len(target_seq):
            if q != '-' and t != '-' :
                query_pos[q_i] = i
                target_pos[t_i] = i
            
            if q == '-' or q == '.':
                t_i += 1

            elif t == '-' or t == '.':
                q_i += 1

            else:
                q_i += 1
                t_i += 1

        query_seq_length = len(query_seq)
        target_seq_length = len(target_seq)
        offset = self.max_offset
        if self.use_random_seq_length:
            query_seq_length = np.random.choice(self.random_seq_lengths, p=self.random_seq_dist)
            target_seq_length = np.random.choice(self.random_seq_lengths, p=self.random_seq_dist)
            
            q_offset = int(self.offset_ratio * (query_seq_length))
            t_offset = int(self.offset_ratio * (target_seq_length))

            offset = min(q_offset, t_offset)

        # Choose random offsets
        qp = query_start
        tp = target_start
        query_offset = random.randint(0, offset) - offset // 2
        query_start = max(0, min(query_start + offset, len(query_seq) - self.seq_length))

        query_relative_start = query_start - qp

        target_offset = random.randint(0, offset) - offset // 2
        target_relative_start = query_relative_start + target_offset

        target_start = max(0, min(target_start + target_relative_start, len(target_seq) - self.seq_length))
        
 
        # Slice sequences
        query_seq = query_seq[query_start:query_start + self.seq_length]
        query_mask = query_mask[query_start:query_start + self.seq_length]
        target_seq = target_seq[target_start:target_start + self.seq_length]
        target_mask = target_mask[target_start:target_start + self.seq_length]
        query_pos = query_pos[query_start:query_start + self.seq_length]
        target_pos = target_pos[target_start:target_start + self.seq_length]

        if query_seq_length < len(query_seq):
            query_seq[query_seq_length:] = 0
            query_mask[query_seq_length:] = False
            query_pos[query_seq_length:] = -1.0
        if target_seq_length < len(target_seq):
            target_seq[target_seq_length:] = 0
            target_mask[target_seq_length:] = False
            target_pos[target_seq_length:] = -2.0
        # Test if we want to swap query/target sequence
        if random.random() < 0.5:
            query_seq, target_seq = target_seq, query_seq
            query_mask, target_mask = target_mask, query_mask
            query_pos, target_pos = target_pos, query_pos


        if self.random_mask_rate > 0.0:
            r = torch.rand(*query_seq.shape)
            query_seq[r < self.random_mask_rate] = 0
            r = torch.rand(*target_seq.shape)
            target_seq[r < self.random_mask_rate] = 0

        # Create target matrix
        target_matrix = (query_pos[:,None] == target_pos[None,:]).float()
        mask_matrix = query_mask[:,None] & target_mask[None,:]

        if torch.sum(target_matrix * mask_matrix) < self.min_aligned_tokens:
            mask_matrix = torch.logical_and(mask_matrix, torch.logical_not(target_matrix))
        
        return target_matrix, query_seq, target_seq, mask_matrix, query_mask, target_mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Target matrix and mask matrix.
        """
        
        file_idx = self.idx_to_file[idx].item()
        alignment_file, query_file, target_file = self.file_list[file_idx]
        # Load files
        try:
            alignment_data = self._load_alignment_file(alignment_file)
            query_seq_data = self._load_seq_file(query_file)
            target_seq_data = self._load_seq_file(target_file)
        except Exception as e:
            print(f"Error loading file {alignment_file}, {query_file}, {target_file}")
            raise e

        # Select alignment
        idx = len(alignment_data) % len(alignment_data)
        alignment = alignment_data[idx]
        query_name = alignment[0]
        target_name = alignment[3]

        # Get the sequences corresponding to the alignment
        query_seq, query_mask = self._create_sequence_tensors(query_seq_data[query_name])
        target_seq, target_mask = self._create_sequence_tensors(target_seq_data[target_name])

        # Create and return training matrices
        return self._create_training_data(alignment, query_seq, query_mask, target_seq, target_mask)

