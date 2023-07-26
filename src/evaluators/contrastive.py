"""" Evaluator class for the contrastive CNN model """

import logging
from typing import List
import time
import faiss
import numpy as np
import torch
import pdb
from src.evaluators.uniref_evaluator import UniRefEvaluator
from src.evaluators.contrastive_functional import filter_scores
from src.utils import create_faiss_index, encode_string_sequence

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
            model_class(
                encode_string_sequence(sequence).unsqueeze(0).to(self.model_device)
            )
            .squeeze()
            .T
        )

    def _setup_targets_for_search(
        self,
        target_embeddings: List[torch.Tensor],
        target_names: List[str],
        lengths: List[int],
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
            device=self.index_device,
            num_threads=self.omp_num_threads,
        )

        self.index.nprobe = self.nprobe

        logger.info("Adding targets to index.")
        if self.index_device == "cpu":
            self.index.add(unrolled_targets.to("cpu"))
        else:
            self.index.add(unrolled_targets)

        faiss.omp_set_num_threads(self.omp_num_threads)

    # def search(self, query_embedding: torch.Tensor):
    #     """Searches through the target DB and gathers a
    #     filtered list of sequences and distances to their centre
    #     which we use as hits for the given query"""

    #     search_start = time.time()

    #     scores_array, indices_array = self.index.search(query_embedding.contiguous(), k=1000)
    #     search_time = time.time() - search_start
    #     filtration_time = time.time()

    #     filtered_scores = filter_scores(
    #         scores_array.to("cpu").numpy(),
    #         indices_array.to("cpu").numpy(),
    #         self.unrolled_names,
    #     )
    #     filtration_time = time.time() - filtration_time

    #     return filtered_scores, search_time, filtration_time
