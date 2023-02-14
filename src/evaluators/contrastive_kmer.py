"""" Evaluator class for the contrastive CNN model """

import os
import logging
import faiss
import numpy as np
from typing import List, Tuple
import torch
from src.evaluators.contrastive import ContrastiveEvaluator
from src.utils import create_faiss_index, encode_string_sequence
import pdb
from torch import nn

logger = logging.getLogger("evaluate")


class ContrastiveKmerEvaluator(ContrastiveEvaluator):
    """Evaluator for the Contrastive Loss CNN model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unrolled_names = []
        self.index: faiss.Index = None
        self.embedding_dimension = 256

        self.num_random_matrices = 1
        self.random_matrices = [torch.randn(size=(self.embedding_dimension, self.embedding_dimension)) for i in range(self.num_random_matrices)]

        self.W = 10

    def transform(self, sequence_embedding: torch.Tensor):
        transformed_sequence_embeddings = []

        for amino_embedding in sequence_embedding:
            transformed_amino_embeddings = []
            normalized_embedding = nn.InstanceNorm1d(amino_embedding)
            for random_matrix in self.random_matrices:
                transformed_embedding = torch.mm(random_matrix, normalized_embedding)
                transformed_amino_embeddings.append(transformed_embedding)

            transformed_sequence_embeddings.append(transformed_amino_embeddings)
        return transformed_sequence_embeddings 

    def find_maximizer(self, transformed_sequence_embedding):


                    
    


