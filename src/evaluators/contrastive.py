from src.evaluators.uniref_evaluator import UniRefEvaluator
import logging
import torch
import faiss
from src.utils import create_faiss_index, encode_string_sequence
import numpy as np
import os
import pdb

logger = logging.getLogger(__file__)


class ContrastiveEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_embedding(self, sequence: str, model_class) -> torch.Tensor:
        return (
            model_class(
                encode_string_sequence(sequence)
                .unsqueeze(0)
                .to(self.model_device)
            )
            .squeeze()
            .T
        )

    def _setup_target_and_query_dbs(
        self, targets, queries, target_names, query_names
    ):
        # no queries?
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(
            zip(lengths, target_names, targets)
        ):
            # sample every N amino.
            aminos = torch.cat(
                [target[j].unsqueeze(0) for j in range(length)], dim=0
            )

            self.unrolled_names.extend(
                [name] * length
            )  # record keeping (num targets x amino per target) - every given amino in a sequence has the same name
            unrolled_targets.append(aminos)

        unrolled_targets = torch.cat(
            unrolled_targets, dim=0
        )  # 128 x (num targets x amino per target)

        logger.info(
            f"Number of aminos in target DB: {unrolled_targets.shape[0]}"
        )

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(
                unrolled_targets, dim=-1
            )

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

    def search(self, query_embedding):
        filtered_list = []

        D, I = self.index.search(
            query_embedding.contiguous(), k=1000
        )  # top 2048 hits
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))
        # TODO: use a torch.cat instead
        # see line 313 in uniref_evaluator
        return filtered_list
