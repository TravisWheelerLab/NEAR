"""" Evaluator class for the contrastive CNN model """

import itertools
import logging
import os
import pdb
from typing import List, Tuple

import faiss
import numpy as np
import torch
from torch import nn

from src.evaluators.contrastive import ContrastiveEvaluator
from src.utils import create_faiss_index, encode_string_sequence

logger = logging.getLogger("evaluate")


class ContrastiveKmerEvaluator(ContrastiveEvaluator):
    """Evaluator for the Contrastive Loss CNN model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unrolled_names = []
        self.index: faiss.Index = None
        self.embedding_dimension = 256

        self.num_random_matrices = 1
        self.random_matrices = [
            torch.randn(size=(self.embedding_dimension, self.embedding_dimension))
            for i in range(self.num_random_matrices)
        ]

        self.W = 10
        self.step_size = 1

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

    def find_maximizers(self, transformed_amino_embedding: torch.Tensor):
        """
        Input: one transformed amino embedding of size embedding dimension"""
        windowed_embedding = transformed_amino_embedding.unfold(
            0, self.W, self.step_size
        )  # num_windows, W
        indices = [i for i in itertools.product(range(self.W), range(self.W)) if i[0] != i[1]]

        maximizers = []
        products = []

        for window in windowed_embedding:
            max_prod = 0
            optimal_pair = None
            # all versus all products
            for pair in indices:
                prod = torch.dot(window[:, pair[0]], window[:, pair[1]])
                if prod > max_prod:
                    max_prod = prod
                    optimal_pair = pair
            maximizers.append(optimal_pair)
            products.append(max_prod)

        return products, maximizers
