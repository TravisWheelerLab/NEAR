import logging
import os
import pdb
from typing import List, Tuple

import scann
import numpy as np
import torch

from src.evaluators.uniref_evaluator import UniRefEvaluator
from src.utils import encode_string_sequence

logger = logging.getLogger("evaluate")


class ContrastiveEvaluatorScaNN(UniRefEvaluator):
    """Evaluator for the Contrastive Loss Model"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.unrolled_names = []

    def compute_embedding(self, sequence: str, model_class) -> torch.Tensor:
        """Encodes the input sequence as a tensor and then passes
        through the model forward function to get the embedding tensor"""

        return (
            model_class(encode_string_sequence(sequence).unsqueeze(0).to(self.model_device))
            .squeeze()
            .T
        )

    def _setup_targets_for_search(
        self, target_embeddings: List[torch.Tensor], target_names: List[str],
    ):
        """Creates the Faiss Index object using the unrolled
        target embddings"""

        lengths: List[int] = list(map(lambda s: s.shape[0], target_embeddings))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []

        for length, name, target in zip(lengths, target_names, target_embeddings):
            # sample every N amino.
            aminos = torch.cat([target[j].unsqueeze(0) for j in range(length)], dim=0)

            self.unrolled_names.extend(
                [name] * length
            )  # record keeping (num targets x amino per target)
            # - every given amino in a sequence has the same name
            unrolled_targets.append(aminos)

        unrolled_targets = torch.cat(
            unrolled_targets, dim=0
        )  # (num targets x amino per target) x 256

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        n = unrolled_targets.shape[0]

        num_partitions = int(np.sqrt(n))
        print(f"Number of targets: {n}")
        print(f"Number of leaves: {num_partitions}")

        self.searcher = (
            scann.scann_ops_pybind.builder(unrolled_targets, 10, "dot_product")
            .tree(
                num_leaves=num_partitions,
                num_leaves_to_search=num_partitions // 10,
                training_sample_size=250000,
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(100)
            .build()
        )

        self.unrolled_names = np.asarray(self.unrolled_names)

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
            try:
                indices += list(indices_array[idx][sorted_idx][unique_indices])
                scores += list(scores_idx[sorted_idx][unique_indices])
            except ValueError as e:
                print(e)
                pdb.set_trace()
        return scores, indices

    def search(self, query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """Searches through the target DB and gathers a
        filtered list of sequences and distances to their centre
        which we use as hits for the given query"""
        filtered_scores = {}

        indices_array, scores_array = self.searcher.search_batched(query_embedding.contiguous())
        # remove stuff that's under/over the threshold

        """ BASED ON MY UNDERSTANDING 
        This should be a matrix 
        and have values for distances for each amino acid in the query sequence """
        # indices = indices_array[self.comp_func(distances_array, self.distance_threshold)]
        # distances = distances_array[self.comp_func(distances_array, self.distance_threshold)] #this has shape sequence length x 1000
        # for each amino, the 1000 target aminos that are closest to that amino
        scores, indices = self.filter_scores(scores_array, indices_array)

        # for distance, name in zip(
        #     distances.ravel().to("cpu").numpy(),
        #     self.unrolled_names[indices.ravel().to("cpu").numpy()],
        # ):
        for distance, name in zip(scores, self.unrolled_names[indices],):
            # filtered_list.append((name, distance))
            if name in filtered_scores.keys():
                filtered_scores[name] += distance
            else:
                filtered_scores[name] = distance
        return filtered_scores  # , filtered_list
