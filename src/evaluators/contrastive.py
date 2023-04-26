"""" Evaluator class for the contrastive CNN model """

import logging
import os
import pdb
from typing import List, Tuple

import faiss
import numpy as np
import torch

from src.evaluators.uniref_evaluator import UniRefEvaluator
from src.utils import create_faiss_index, encode_string_sequence
import time

logger = logging.getLogger("evaluate")


class ContrastiveEvaluator(UniRefEvaluator):
    """Evaluator for the Contrastive Loss Model"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.index: faiss.Index = None

    def compute_embedding(self, sequence: str, model_class) -> torch.Tensor:
        """Encodes the input sequence as a tensor and then passes
        through the model forward function to get the embedding tensor"""

        return (
            model_class(encode_string_sequence(sequence).unsqueeze(0).to(self.model_device))
            .squeeze()
            .T
        )

    def _setup_targets_for_search(
        self, target_embeddings: List[torch.Tensor], target_names: List[str], lengths: List[int]
    ):
        """Creates the Faiss Index object using the unrolled
        target embddings"""

        self.unrolled_names = np.repeat(target_names, lengths)
        unrolled_targets = torch.cat(
            target_embeddings, dim=0
        )  # (num targets x amino per target) x 256

        logger.info(f"Original DB size: {sum(lengths)}")
        del lengths

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index: faiss.Index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,  # f"IVF{K},PQ8", #self.index_string, #f"IVF100,PQ8", #"IndexIVFFlat", #self.index_string,
            nprobe=self.nprobe,
            device=self.index_device,
        )

        logger.info("Adding targets to index.")
        if self.index_device == "cpu":
            self.index.add(unrolled_targets.to("cpu"))
        else:
            self.index.add(unrolled_targets)

        faiss.omp_set_num_threads(self.num_threads)

    def filter_scores(self, scores_array, indices_array):
        """Filters the scores such that every query amino can only
        be matched to one amino from each target sequence
        and it matches the one with the biggest score"""
        scores = []
        indices = []

        for idx in range(len(scores_array)):
            scores_idx = scores_array[idx]
            names = self.unrolled_names[
                indices_array[idx]
            ]  # the names of the targets for each 1000 hits
            sorted_idx = np.argsort(scores_idx)[::-1]

            _, unique_indices = np.unique(
                names[sorted_idx], return_index=True
            )  # the unique names of the targets for each 1000 hits (<= 1000)
            indices += list(indices_array[idx][sorted_idx][unique_indices])
            scores += list(scores_idx[sorted_idx][unique_indices])

        return scores, indices

    def search(self, query_embedding: torch.Tensor):
        """Searches through the target DB and gathers a
        filtered list of sequences and distances to their centre
        which we use as hits for the given query"""
        filtered_scores = {}

        scores_array, indices_array = self.index.search(query_embedding.contiguous(), k=1000)

        scores, indices = self.filter_scores(
            scores_array.to("cpu").numpy(), indices_array.to("cpu").numpy()
        )

        for distance, name in zip(
            scores,
            self.unrolled_names[indices],
        ):
            if name in filtered_scores.keys():
                filtered_scores[name] += distance
            else:
                filtered_scores[name] = distance

        return filtered_scores
