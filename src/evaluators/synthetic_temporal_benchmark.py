import pdb

import torch

from src.evaluators import Evaluator
from src.utils import create_faiss_index


class TemporalBenchmark(Evaluator):
    def __init__(
        self,
        target_db_size,
        num_queries,
        embedding_dimension,
        n_neighbors,
        index_device,
        model_device,
    ):
        super(TemporalBenchmark).__init__()
        self.target_db_size = target_db_size
        self.n_neighbors = n_neighbors
        self.embedding_dimension = embedding_dimension
        self.index_device = index_device
        self.model_device = model_device
        self.num_queries = num_queries

    def evaluate(self, model_class):
        # assume file IO is insigificant
        # create some target db;
        # and normalize.
        random_target_embeddings = torch.randn(
            (self.target_db_size, self.embedding_dimension)
        )
        # normalize;
        random_target_embeddings = torch.nn.functional.normalize(
            random_target_embeddings, dim=0
        )

        index = create_faiss_index(
            random_target_embeddings,
            self.embedding_dimension,
            "Flat",
            device=self.index_device,
        )
        pdb.set_trace()
