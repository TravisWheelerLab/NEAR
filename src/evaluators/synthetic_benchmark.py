import logging
import pdb
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from src.datasets.datasets import sanitize_sequence
from src.evaluators import Evaluator
from src.utils import (
    amino_distribution,
    amino_n_to_a,
    create_substitution_distribution,
    encode_string_sequence,
    encode_tensor_sequence,
    fasta_from_file,
)
from src.utils.gen_utils import amino_alphabet, mutate_sequence
from src.utils.helpers import create_faiss_index

logger = logging.getLogger("evaluate")


class SyntheticEvaluator(Evaluator):
    def __init__(
        self,
        target_sequence_fasta,
        num_queries,
        blosum,
        device,
        sample_percent,
        normalize_embeddings,
        index_string,
        index_device,
        query_percent,
        distance_threshold,
    ):
        self.blosum = blosum
        if blosum not in [45, 62, 80, 90]:
            raise ValueError("blosum must be one of <45, 62, 80, 90>")

        self.blosum_file = f"src/resources/blosum{blosum}.probs"
        self.target_sequence_fasta = target_sequence_fasta
        self.num_queries = num_queries
        self.index_string = index_string
        self.index_device = index_device
        self.sample_percent = sample_percent
        self.distance_threshold = distance_threshold
        self.query_percent = query_percent
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.aa_dist = amino_distribution
        self.sub_dists = self._sub_dists()

        if self.normalize_embeddings:
            logger.info("Using comparison function >= threshold for filtration.")
            self.comp_func = torch.greater_equal
        else:
            logger.info("Using comparison function <= threshold for filtration.")
            self.comp_func = torch.less_equal

    def _sub_dists(self):
        return create_substitution_distribution(self.blosum)

    def compute_embedding(self, sequence, model_class):
        raise NotImplementedError()

    def create_target_and_query_dbs(self, model_class):
        # now go through and mutate a random selection of target sequences
        target_names, target_sequences = fasta_from_file(self.target_sequence_fasta)
        self.num_target_sequences = len(target_sequences)
        # mutate the query templates
        torch.manual_seed(0)
        np.random.seed(0)
        shuf_idx = torch.randperm(len(target_names))[: self.num_queries]
        queries = []
        query_names = []
        logger.debug("Starting computation of databases.")

        target_names = torch.arange(len(target_names))

        for shuffled_idx in shuf_idx:
            # target_sequence = sanitize_sequence(target_sequences[shuffled_idx])
            target_sequence = target_sequences[shuffled_idx]
            encoding = encode_string_sequence(target_sequence).argmax(dim=0)
            # ok, the encoding is working to reconstruct.
            # found the issue.
            # tensor clone goes in here to create the mutated sequence.
            # is there an issue here?
            mutated = mutate_sequence(
                sequence=encoding,
                substitutions=len(target_sequences[shuffled_idx]),
                sub_distributions=self.sub_dists,
            )
            queries.append([amino_alphabet[i.item()] for i in mutated])
            query_names.append(target_names[shuffled_idx])

        # compute embeddings
        # forget about batching
        target_embeddings = self.calc_embeddings(target_sequences, model_class)
        query_embeddings = self.calc_embeddings(queries, model_class)

        # the target and query embeddings are _not_ the same.
        # for t in target_embeddings:
        #     for q in query_embeddings:
        #         if torch.all(q == t):
        #             pdb.set_trace()
        # the loop above does not enter pdb.
        # unroll target embeddings

        lengths = list(map(lambda s: s.shape[0], target_embeddings))
        logger.info(f"Original DB size: {sum(lengths)}")

        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(
            zip(lengths, target_names, target_embeddings)
        ):
            if self.sample_percent != 1.0:
                n_sample = length * self.sample_percent
                # sample every N amino.
                sampled_idx = torch.randperm(length)[: int(length / n_sample)]
            else:
                sampled_idx = torch.arange(length)

            if len(sampled_idx) == 0:
                sampled_idx = [0]

            sampled_aminos = torch.cat(
                [target[j].unsqueeze(0) for j in sampled_idx], dim=0
            )
            self.unrolled_names.extend([name] * len(sampled_aminos))
            unrolled_targets.append(sampled_aminos)

            logger.debug(
                f"sampled_index length: {len(sampled_idx)}. original length: {length}"
            )

        unrolled_targets = torch.cat(unrolled_targets, dim=0)
        assert len(unrolled_targets) == len(self.unrolled_names)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            nprobe=1,
            device=self.index_device,
        )
        self.unrolled_names = np.asarray(self.unrolled_names)

        logger.info("Adding targets to index.")
        self.index.add(unrolled_targets)

        return query_embeddings, query_names

    def filter(
        self,
        queries,
        query_names,
        start=0,
        end=torch.inf,
    ):
        qdict = dict()

        logger.info("Beginning search.")

        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0
        t_begin = time.time()

        logger.info(f"Selecting {self.query_percent * 100}% of query aminos.")

        for i in range(start, num_queries):
            loop_begin = time.time()
            logger.debug(f"{i / (num_queries - start):.3f}")

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

            qval = qval[
                torch.randperm(qval.shape[0])[: int(self.query_percent * qval.shape[0])]
            ]
            filtered_hits = self.search(qval)
            qdict[query_names[i]] = filtered_hits
            time_taken = time.time() - loop_begin
            t_tot += time_taken

            logger.debug(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        return qdict, loop_time / i, loop_time

    def calc_embeddings(self, sequences, model_class):
        embeddings = []
        for sequence in sequences:
            embed = self.compute_embedding(sequence, model_class)
            embeddings.append(embed)
        return embeddings

    @torch.no_grad()
    def evaluate(self, model_class):
        query_embeddings, query_names = self.create_target_and_query_dbs(model_class)
        hits, avg_it, total_t = self.filter(query_embeddings, query_names)
        # now just compute recall.
        # this is easy.
        recall = 0
        total_hits = 0
        # I'm satisfied with this.
        for query, hitset in hits.items():
            distances = [h[1] for h in hitset]
            hit_labels = np.asarray([h[0] for h in hitset])
            hit_labels = hit_labels[np.argsort(distances)[::-1]]
            # recall and filtration.
            if query.item() in hit_labels:
                recall += 1
            total_hits += len(hit_labels)

        logger.info(
            f"num queries: {len(hits)}. Sequences in target DB: {self.num_target_sequences}."
        )
        desc = f"blosum {self.blosum}/"
        if model_class.apply_cnn_loss:
            desc += "cnn loss/"
        if model_class.backprop_on_near_aminos:
            desc += "threshold loss distance/"
        if model_class.apply_contrastive_loss:
            desc += "supcon loss/"
        desc += f"{model_class.downsample_steps}/"
        desc += f"{self.distance_threshold}"

        print("model description:", desc)
        print(
            f"recall {(recall / len(hits)) * 100}%. Filtration: {(1 - (total_hits / (self.num_queries * self.num_target_sequences))) * 100:.3f}%"
        )

    def search(self, query_embedding):
        raise NotImplementedError()


class SyntheticVAEEvaluator(SyntheticEvaluator):
    def compute_embedding(self, sequence, model_class):
        slices = []
        for i in range(
            0, len(sequence) - model_class.initial_seq_len, model_class.initial_seq_len
        ):
            embedding, _ = model_class(
                encode_string_sequence(sequence[i : i + model_class.initial_seq_len])
                .to(self.device)
                .unsqueeze(0)
            )
            slices.append(embedding)

        if len(sequence) % model_class.initial_seq_len != 0:
            embedding, _ = model_class(
                encode_string_sequence(sequence[-model_class.initial_seq_len :])
                .to(self.device)
                .unsqueeze(0)
            )
            slices.append(embedding)

        if len(sequence) == model_class.initial_seq_len:
            embedding, _ = model_class(
                encode_string_sequence(sequence).to(self.device).unsqueeze(0)
            )
            slices.append(embedding)

        return torch.cat(slices, dim=-1).squeeze().T

    def search(self, query_embedding):
        filtered_list = []

        D, I = self.index.search(query_embedding.contiguous(), k=2048)
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        # sort distance
        # get the unique indices (unique_idx returns the first occurence
        unique, unique_idx = np.unique(I.to("cpu").numpy().ravel(), return_index=True)
        # now get unique names
        # subsample D
        unique_distances = D.to("cpu").numpy().ravel()[unique_idx]

        unique_names, unique_name_idx = np.unique(
            self.unrolled_names[unique], return_index=True
        )

        unique_distances = unique_distances[unique_name_idx]

        if D.numel() > 0:
            logger.debug(f"min: {torch.min(D)}, max: {torch.max(D)}")
            logger.debug(
                f"Number of matches at a distance threshold of {self.distance_threshold}: {unique_names.shape}."
                f" Because of faiss limitations, actual distance threshold achieved is {torch.min(D)}"
            )

        for distance, name in zip(unique_distances.ravel(), unique_names.ravel()):
            filtered_list.append((name, distance.item()))

        return filtered_list
