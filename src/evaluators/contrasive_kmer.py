"""" Evaluator class for the contrastive CNN model 
This evaluator attempts to reduce the embedding of the sequence using
a KMER like algorithm and perform nearest neighbor search on the reduced space"""

import itertools
import logging

import faiss
import torch
from torch import nn
from typing import Tuple, List
import tqdm
from src.evaluators.contrastive import ContrastiveEvaluator

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
        ]  # should these be between 0 and 1 since we just normalized? yes they are random normal...

        self.W = 10
        self.step_size = 1

    def transform(self, sequence_embeddings: List[torch.Tensor]):
        transformed_sequence_embeddings = []

        for sequence_embedding in sequence_embeddings:  # dim = sequence len, amino embedding
            sequence_embedding = sequence_embedding.unsqueeze(1)
            n = nn.InstanceNorm1d(sequence_embedding.shape[0])
            normalized_embedding = n(sequence_embedding)
            for random_matrix in self.random_matrices:
                transformed_embedding = torch.mm(normalized_embedding.squeeze(1), random_matrix)
                transformed_sequence_embeddings.append(transformed_embedding)
        return transformed_sequence_embeddings  # list of tensors of shape 506,256

    def reduce(self, sequence_embeddings: List[torch.Tensor]):
        """I think right now i'm just gonna put both pairs in without keeping track of
        the fact that they are pairs.. but i should rethink this. when we do search maybe we only want
        to do a distance for each pair

        also maybe want to see whether i can plug 2d vectors into FAISS

        The reduction takes a list of sequence embeddinigs of the embedding dim
        and reduces them by creating windows of length W
        and calculating the cosine similarity of each pair of windows
        and returning the pairs with the smallest similarity"""
        indices = [i for i in itertools.product(range(self.W), range(self.W)) if i[0] != i[1]]

        sequence_minimizers = []
        for embedding in tqdm.tqdm(sequence_embeddings):
            windowed_embedding = embedding.unfold(0, self.W, self.step_size)
            minimizers = torch.Tensor(len(windowed_embedding), 2, windowed_embedding[0].shape[0])
            for idx, window in enumerate(windowed_embedding):
                optimal_pair = None
                min_sim = 10000
                for pair in indices:
                    A = window[:, pair[0]]
                    B = window[:, pair[1]]
                    cos_sim = torch.dot(A, B) / (torch.norm(A) * torch.norm(B))
                    if torch.abs(cos_sim) < min_sim:
                        min_sim = cos_sim
                        optimal_pair = torch.stack((A, B))
                minimizers[idx] = optimal_pair
            sequence_minimizers.append(
                minimizers.reshape(len(windowed_embedding) * 2, windowed_embedding[0].shape[0])
            )

        return sequence_minimizers

    @torch.no_grad()
    def _calc_embeddings(
        self,
        sequence_data: dict,
        model_class,
        apply_random_sequence: bool,
        max_seq_length=512,
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Calculates the embeddings for the sequences by
        calling the model forward function. Filters the sequences by max/min
        sequence length and returns the filtered sequences/names and embeddings

        Returns [names], [sequences], [embeddings]"""

        names = list(sequence_data.keys())
        sequences = list(sequence_data.values())

        logger.info("Filtering sequences by length...")
        filtered_names, embeddings, lengths = self.filter_sequences_by_length(
            names,
            sequences,
            model_class,
            apply_random_sequence,
            max_seq_length,
        )

        assert len(filtered_names) == len(embeddings)

        print("Reducing...")
        embeddings_minimized = self.reduce(embeddings)

        return filtered_names, embeddings_minimized, lengths

    def evaluate(self, model_class) -> dict:
        """Evaluation pipeline.

        Calculates embeddings for query and targets
        If visactmaps is true, generates activation map plots given the target embeddings
            (probably want to remove this feature)
        Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
        """
        if hasattr(model_class, "initial_seq_len"):
            self.tile_size = model_class.initial_seq_len

        print(f"Found {(len(self.target_seqs))} targets")

        print("Embedding queries...")
        query_names, query_embeddings, _ = self._calc_embeddings(
            sequence_data=self.query_seqs,
            model_class=model_class,
            apply_random_sequence=self.add_random_sequence,
            max_seq_length=self.max_seq_length,
        )
        del self.query_seqs

        print("Embedding targets...")
        target_names, target_embeddings, target_lengths = self._calc_embeddings(
            sequence_data=self.target_seqs,
            model_class=model_class,
            apply_random_sequence=False,
            max_seq_length=self.max_seq_length,
        )

        del self.target_seqs  # remove from memory

        self._setup_targets_for_search(target_embeddings, target_names, target_lengths)

        model_hits, _, _ = self.filter(query_embeddings, query_names)

        return model_hits
