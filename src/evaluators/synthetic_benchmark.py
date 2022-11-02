import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

warnings.filterwarnings("ignore")

from src.evaluators import Evaluator
from src.utils import (
    amino_distribution,
    create_substitution_distribution,
    encode_string_sequence,
    fasta_from_file,
)
from src.utils.gen_utils import (
    amino_alphabet,
    generate_string_sequence,
    mutate_sequence,
)
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
        figure_path,
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
        self.figure_path = figure_path
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

        for i in tqdm.tqdm(range(start, num_queries)):
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
        for sequence in tqdm.tqdm(sequences):
            embed = self.compute_embedding(sequence, model_class)
            embeddings.append(embed)
        return embeddings

    def search(self, query_embedding):
        raise NotImplementedError()

    @torch.no_grad()
    def evaluate(self, model_class):
        query_embeddings, query_names = self.create_target_and_query_dbs(model_class)
        hits, avg_it, total_t = self.filter(query_embeddings, query_names)

        # now just compute recall.
        # this is easy.

        threshold_to_recall = {}
        threshold_to_total_hits = {}

        for threshold in np.linspace(self.distance_threshold, 1.0, num=30):
            logger.debug(f"threshold: {threshold}")
            recall = 0
            total_hits = 0

            for query, hitset in hits.items():
                distances = np.array([h[1] for h in hitset])
                hit_labels = np.asarray([h[0] for h in hitset])
                # remove labels under/over the threshold
                if self.normalize_embeddings:
                    hit_labels_filtered = hit_labels[
                        np.where(distances >= threshold)[0]
                    ]
                else:
                    hit_labels_filtered = hit_labels[
                        np.where(distances <= threshold)[0]
                    ]
                # recall and filtration.
                if isinstance(query, int):
                    if query in hit_labels_filtered:
                        recall += 1
                else:
                    if query.item() in hit_labels_filtered:
                        recall += 1

                # only get uniques
                total_hits += len(set(hit_labels_filtered))

            threshold_to_total_hits[threshold] = total_hits
            threshold_to_recall[threshold] = recall

        logger.info(
            f"num queries: {len(hits)}. Sequences in target DB: {self.num_target_sequences}."
        )

        recalls = list(threshold_to_recall.values())
        total_hits = list(threshold_to_total_hits.values())
        filtrations = [
            100 * (1 - (t / (self.num_target_sequences * self.num_target_sequences)))
            for t in total_hits
        ]
        recalls = [100 * (r / len(hits)) for r in recalls]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(self.figure_path)
        ax.scatter(filtrations, recalls, c="r", marker="o")
        ax.plot(filtrations, recalls, "r--", linewidth=2)
        ax.set_ylim([0, 101])
        ax.set_xlim([0, 101])
        plt.savefig(self.figure_path)
        plt.close()


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

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))

        return filtered_list


class KMerEmbedEvaluator(SyntheticEvaluator):
    def __init__(self, num_targets, kmer_length, *args, seq_len=128, **kwargs):
        super(KMerEmbedEvaluator, self).__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.num_targets = num_targets
        self.num_target_sequences = num_targets
        self.kmer_length = kmer_length
        self.index = None

    def compute_embedding(self, sequence, model_class):
        return model_class(
            encode_string_sequence(sequence).to(self.device).unsqueeze(0)
        ).mean(dim=-1)

    def search(self, query_embedding):
        filtered_list = []

        D, I = self.index.search(query_embedding.contiguous(), k=2048)
        # remove stuff that's under/over the threshold
        I = I[self.comp_func(D, self.distance_threshold)]
        D = D[self.comp_func(D, self.distance_threshold)]

        for distance, name in zip(
            D.ravel().to("cpu").numpy(),
            self.unrolled_names[I.ravel().to("cpu").numpy()],
        ):
            filtered_list.append((name, distance))

        return filtered_list

    def create_target_and_query_dbs(self, model_class):
        torch.manual_seed(0)
        np.random.seed(0)
        queries = []
        query_names = []
        target_names = []
        target_sequences = []
        logger.debug("Starting computation of databases.")

        for i in range(self.num_targets):
            target_sequence = generate_string_sequence(self.seq_len)
            start_idx = int(np.random.rand() * (self.seq_len - self.kmer_length))
            # make a kmer seed
            kmer_seed = target_sequence[start_idx : start_idx + self.kmer_length]
            random_seq = generate_string_sequence(self.seq_len)
            start_idx = int(np.random.rand() * (self.seq_len - self.kmer_length))
            seeded_seq = (
                random_seq[:start_idx]
                + kmer_seed
                + random_seq[start_idx + self.kmer_length :]
            )

            queries.append(seeded_seq)
            target_sequences.append(target_sequence)

            query_names.append(i)
            target_names.append(i)

        target_embeddings = self.calc_embeddings(target_sequences, model_class)
        query_embeddings = self.calc_embeddings(queries, model_class)

        lengths = list(map(lambda s: s.shape[0], target_embeddings))
        logger.info(f"Original DB size: {sum(lengths)}")

        unrolled_targets = []
        self.unrolled_names = []

        for i, (length, name, target) in enumerate(
            zip(lengths, target_names, target_embeddings)
        ):
            self.unrolled_names.extend([name] * len(target))
            unrolled_targets.append(target)

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
