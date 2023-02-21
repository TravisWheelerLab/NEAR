"""" Evaluator class for the contrastive CNN model """

import os
import logging
import faiss
import numpy as np
from typing import List, Tuple
import torch
from src.evaluators.uniref_evaluator import UniRefEvaluator
from src.utils import create_faiss_index, encode_string_sequence
import pdb

logger = logging.getLogger("evaluate")


class ContrastiveEvaluator(UniRefEvaluator):
    """Evaluator for the Contrastive Loss CNN model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unrolled_names = []
        self.index: faiss.Index = None

    def compute_embedding(self, sequence: str, model_class) -> torch.Tensor:
        """Encodes the input sequence as a tensor and then passes
        through the model forward function to get the embedding tensor"""
        return (
            model_class(encode_string_sequence(sequence).unsqueeze(0).to(self.model_device))
            .squeeze()
            .T
        )

    def _setup_targets_for_faiss(
        self,
        target_embeddings: List[torch.Tensor],
        target_names: List[str],
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

        self.index: faiss.Index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            nprobe=self.nprobe,
            device=self.index_device,
        )

        logger.info("Adding targets to index.")
        if self.index_device == "cpu":
            self.index.add(unrolled_targets.to("cpu"))
        else:
            self.index.add(unrolled_targets)

        self.unrolled_names = np.asarray(self.unrolled_names)

        faiss.omp_set_num_threads(int(os.environ.get("NUM_THREADS")))

    def filter_scores(self, scores_array, indices_array):
        """Filters the scores such that every query amino can only
        be matched to one amino from each target sequence
        and it matches the one with the biggest score"""
        scores = []
        indices = []
        
        for idx in range(len(scores_array)):
            scores = scores_array[idx]
            names = self.unrolled_names[indices_array[idx]] #the names of the targets for each 1000 hits
            sorted_idx = np.argsort(scores)[::-1]

            _, unique_indices = np.unique(names[sorted_idx], return_index=True) #the unique names of the targets for each 1000 hits (<= 1000)
            try:
                indices += indices_array[idx][sorted_idx][unique_indices]
                scores += scores[sorted_idx][unique_indices]
            except ValueError as e:
                print(e)
                pdb.set_trace()
        return scores, indices


    def search(self, query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """Searches through the target DB and gathers a
        filtered list of sequences and distances to their centre
        which we use as hits for the given query"""
        filtered_list = []
        filtered_scores = {}

        scores_array, indices_array = self.index.search(query_embedding.contiguous(), k=1000)
        # remove stuff that's under/over the threshold

        """ BASED ON MY UNDERSTANDING 
        This should be a matrix 
        and have values for distances for each amino acid in the query sequence """
        #indices = indices_array[self.comp_func(distances_array, self.distance_threshold)]
        #distances = distances_array[self.comp_func(distances_array, self.distance_threshold)] #this has shape sequence length x 1000
        #for each amino, the 1000 target aminos that are closest to that amino
        scores, indices = self.filter_scores(scores_array.to("cpu").numpy(), indices_array.to("cpu").numpy())

        # for distance, name in zip(
        #     distances.ravel().to("cpu").numpy(),
        #     self.unrolled_names[indices.ravel().to("cpu").numpy()],
        # ):
        for distance, name in zip(
            scores,
            self.unrolled_names[indices],
        ):
            #filtered_list.append((name, distance))
            if name in filtered_scores.keys():
                filtered_scores[name] += distance
            else:
                filtered_scores[name] = distance

        return filtered_scores#, filtered_list
