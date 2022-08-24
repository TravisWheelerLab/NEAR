import pdb
from collections import defaultdict

import numpy as np
import torch

from src.datasets.datasets import ClusterIterator
from src.evaluators import Evaluator
from src.utils import (
    compute_cluster_representative_embeddings,
    create_faiss_index,
    most_common_matches,
)


class AccuracyComputer(Evaluator):
    def __init__(
        self,
        fasta_files,
        sequence_length,
        include_all_families,
        n_seq_per_target_family,
        normalize,
        embed_dim,
        n_neighbors,
        batch_size,
        collate_fn,
        quantize_index,
        device,
    ):

        self.normalize = normalize
        self.n_neighbors = n_neighbors
        self.device = device
        self.embed_dim = embed_dim
        self.quantize_index = quantize_index
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        self.iterator = ClusterIterator(
            fasta_files,
            sequence_length,
            representative_index=0,
            include_all_families=include_all_families,
            n_seq_per_target_family=n_seq_per_target_family,
        )

    @torch.no_grad()
    def evaluate(self, model_class):

        rep_seqs, rep_labels = self.iterator.get_cluster_representatives()
        rep_gapped_seqs = self.iterator.seed_alignments
        index_device = "cpu"
        # stack the seed sequences.
        rep_embeddings, cluster_rep_labels = compute_cluster_representative_embeddings(
            rep_seqs,
            rep_labels,
            model_class,
            normalize=self.normalize,
            device=self.device,
        )

        cluster_rep_index = create_faiss_index(
            rep_embeddings,
            self.embed_dim,
            device=index_device,
            distance_metric="cosine" if self.normalize else "l2",
            quantize=self.quantize_index,
        )

        query_dataset = torch.utils.data.DataLoader(
            self.iterator,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
        )

        total_sequences = 0
        thresholds = [1, 2, 3, 5, 10, 20, 100, 200]
        topn = defaultdict(int)

        for j, (features, labels, sequences) in enumerate(query_dataset):
            if features.shape[0] == 128:
                features = features.unsqueeze(0)
            embeddings = model_class(features.to(self.device)).transpose(-1, -2)

            print(f"{j / len(query_dataset):.3f}", end="\r")
            # searching each sequence separately against the index is probably slow.
            for label, sequence in zip(labels, embeddings):
                if not isinstance(label, int):
                    label = int(label)
                total_sequences += 1
                if self.normalize:
                    sequence = torch.nn.functional.normalize(
                        sequence, dim=-1
                    ).contiguous()
                else:
                    sequence = sequence.contiguous()

                predicted_labels, counts = most_common_matches(
                    cluster_rep_index,
                    cluster_rep_labels,
                    sequence,
                    self.n_neighbors,
                    index_device,
                )

                top_preds = predicted_labels[np.argsort(counts)]

                for n in thresholds:
                    top_pred = top_preds[-n:]
                    if label in set(top_pred):
                        topn[n] += 1

        correct_counts = np.asarray([topn[t] for t in thresholds])
        thresholds = ", ".join([str(t) for t in thresholds])
        percent_correct = ", ".join(
            [f"{c / total_sequences:.3f}" for c in correct_counts]
        )
        _, total_families = np.unique(cluster_rep_labels, return_counts=True)

        print(
            f"{thresholds}\n",
            f"{percent_correct}\n" f"Total families searched: {len(total_families)}",
            f"Total sequences: {total_sequences}",
        )
