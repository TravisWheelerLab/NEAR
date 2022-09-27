import pdb

import torch

from src.datasets.datasets import SequenceIterator
from src.evaluators import Evaluator
from src.utils import create_faiss_index, encode_string_sequence, fasta_from_file


class TemporalBenchmark(Evaluator):
    def __init__(
        self,
        target_file,
        query_file,
        decoy_file,
        model_device,
        index_device,
        n_neighbors,
        min_sequence_length,
        max_sequence_length,
        query_batch_size,
        query_num_workers,
        index_batch_size,
        distance_threshold,
        index_string="Flat",
        normalize_embeddings=True,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.target_file = target_file
        self.query_file = query_file
        self.decoy_file = decoy_file
        self.model_device = model_device
        self.n_neighbors = n_neighbors
        self.normalize_embeddings = normalize_embeddings
        self.index_string = index_string
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.query_num_workers = query_num_workers
        self.index_device = index_device
        self.index_batch_size = index_batch_size
        self.query_batch_size = query_batch_size
        self.distance_threshold = distance_threshold
        self.index = None
        self.unrolled_target_names = []
        # how should this benchmark work?
        # remove computation of the target database
        # from the time calculation
        # the thing I want to compute is time/it for searching query sequence against the
        # target DB, on CPU and GPU

    def compute_target_db(self, model_class):
        target_names, target_sequences = fasta_from_file(self.target_file)
        embeddings = []

        for name, sequence in zip(target_names, target_sequences):
            if self.max_seq_length >= len(sequence) >= model_class.initial_seq_len:
                embed, _ = self.compute_embedding(sequence, model_class)
                embeddings.append(embed)
                self.unrolled_target_names.extend([name] * len(embed))

        unrolled_targets = torch.cat(embeddings, dim=0)

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            nprobe=None,
            device=self.index_device,
        )

        self.index.add(unrolled_targets)

    def evaluate_query_dataloader(self, query_file, model_class):
        # batch and stuff?
        query_dataset = SequenceIterator(query_file, self.min_sequence_length)
        query_dataloader = torch.utils.data.DataLoader(
            query_dataset,
            batch_size=self.query_batch_size,
            collate_fn=query_dataset.collate_fn(),
            num_workers=self.query_num_workers,
        )
        # it's probably fastest to compute all the embeddings in one go
        query_embeddings = []
        query_names = []
        for features, _, names in query_dataloader:
            query_embeddings.append(model_class(features.to(self.model_device)))
            query_names.extend(names)
        # and then search them against the database
        return query_embeddings, query_names

    def compute_and_search(self, query_sequences, model_class):
        # compute query sequence embedding (batched?)
        # return hits above threshold
        raise NotImplementedError()

    def compute_embedding(self, sequences, model_class):
        if isinstance(sequences, list):
            encoded = encode_string_sequence(sequence).to(self.model_device)
        else:
            encoded = (
                encode_string_sequence(sequence).to(self.model_device).unsqueeze(0)
            )

        embed, _ = model_class(encoded)
        # compute a single embedding or batch of embeddings
        # this creates an index
        return embed

    def evaluate(self, model_class):
        # this creates target names and the index.
        self.create_target_db(model_class)
        query_embeddings, query_names = evaluate_query_dataloader(
            self.query_file, model_class
        )
        pdb.set_trace()
        # now batch the query embeddings and search them against the index
        query_name_to_match = {}
        for i in range(
            0, query_embeddings.shape[0] - self.index_batch_size, self.index_batch_size
        ):
            distances, indices = self.index.search(
                query_embeddings[i : i + self.index_batch_size], k=self.n_neighbors
            )
            indices = indices[distances >= self.distance_threshold] // 4
            # now put the indices into the query names by dividing by the number of elements per
            # embedding (i think it's 4)
            # i need to interactively debug this
