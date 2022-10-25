import logging
import time

import torch

from src.evaluators import Evaluator
from src.models.resnet import ResNet10M, ResNet50M, ResNet100M
from src.utils import create_faiss_index

logger = logging.getLogger("evaluate")

resnet_map = {"10M": ResNet10M, "50M": ResNet50M, "100M": ResNet100M}


class TemporalBenchmark(Evaluator):
    def __init__(
        self,
        target_db_size,
        num_queries,
        embedding_dimension,
        n_neighbors,
        index_device,
        model_device,
        batch_size,
        resnet_n_params,
        sequence_length=340,
    ):
        super(TemporalBenchmark).__init__()
        self.target_db_size = target_db_size
        self.n_neighbors = n_neighbors
        self.embedding_dimension = embedding_dimension
        self.index_device = index_device
        self.model_device = model_device
        self.num_queries = num_queries
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.resnet = resnet_map[resnet_n_params](
            res_block_n_filters=self.embedding_dimension
        ).to(model_device)

    @torch.no_grad()
    def evaluate(self, model_class):
        # create model class here
        #
        del model_class
        logger.info("Beginning evaluation.")
        random_target_embeddings = torch.randn(
            (int(self.target_db_size), self.embedding_dimension)
        )
        # normalize;
        random_target_embeddings = torch.nn.functional.normalize(
            random_target_embeddings, dim=0
        )
        logger.info("Creating index.")
        index = create_faiss_index(
            random_target_embeddings,
            self.embedding_dimension,
            "Flat",
            device=self.index_device,
            nprobe=1,
        )

        logger.info("Adding targets to index.")
        index.add(random_target_embeddings)
        total_params = sum(p.numel() for p in self.resnet.parameters())
        logger.info(f"Model total params: {total_params}.")
        logger.info(
            f"Starting evaluation with {self.num_queries} queries against"
            f" a target database of size {int(self.target_db_size)}. Searching for"
            f" {self.n_neighbors} matches for each sequence."
        )

        begin = time.time()
        cnt = 1
        for i in range(0, self.num_queries, self.batch_size):
            query_input = torch.randn((self.batch_size, 20, self.sequence_length)).to(
                self.model_device
            )
            model_begin = time.time()
            embeddings = self.resnet(query_input)
            logger.debug(f"it: {cnt}, model time/it: {(time.time()-model_begin)}")
            # should be batch size x 128
            # need to do range search
            _, _ = index.search(embeddings, self.n_neighbors)
            logger.debug(f"it: {cnt}, time/it: {(time.time()-begin)/cnt}")
            cnt += 1

        end = time.time()
        logger.info(f"Total time: {end - begin}")
