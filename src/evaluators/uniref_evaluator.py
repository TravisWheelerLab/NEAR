import logging
import os
import pdb
import pickle
import re
import time
from collections import defaultdict
from copy import deepcopy

import faiss
import matplotlib.pyplot as plt
import numba
import numpy as np
import scann
import torch
import tqdm

from src.evaluators import Evaluator
from src.utils import (
    create_faiss_index,
    encode_tensor_sequence,
    fasta_from_file,
    search_index_device_aware,
)

logger = logging.getLogger("evaluate")

# fmt: off
@numba.jit(nopython=True)
def compute_ali_score(dot_products):
    scores = np.zeros_like(dot_products)
    scores[:, 0] = dot_products[:, 0]
    scores[0, :] = dot_products[0, :]

    best_path = np.zeros((scores.shape[0], scores.shape[1], 2), dtype=numba.int64)
    # for each row
    for i in range(1, scores.shape[0]):
        # for each column
        for j in range(1, scores.shape[1]):
            vals = np.array(
                [
                    scores[i - 1, j],
                    scores[i, j - 1],
                    scores[i - 1, j - 1] + dot_products[i, j],
                ]
            )
            idxs = np.array([[i - 1, j], [i, j - 1], [i - 1, j - 1]])
            amax = np.argmax(vals)

            scores[i, j] = vals[amax]
            best_path[i, j] = idxs[amax]

    # best column:
    best_col = np.argmax(scores[-1, :])
    best_row = np.argmax(scores[:, -1])
    if scores[-1, best_col] > scores[best_row, -1]:
        starting_point = best_path[-1, best_col]
    else:
        starting_point = best_path[best_row, -1]

    row_idx = starting_point[0]
    col_idx = starting_point[1]
    # while we haven't reached a side:
    path_log = [starting_point]
    total_score = 0

    while row_idx != 0 and col_idx != 0:
        next_best = best_path[row_idx, col_idx]
        total_score += dot_products[row_idx, col_idx]
        path_log.append(next_best)
        row_idx, col_idx = best_path[row_idx, col_idx]

    return np.abs(total_score)

def filter_hits(queries, threshold, comp_func):
    filtered_queries = dict()
    for query in queries:
        filtered_queries[query] = dict()
        for hit in queries[query]:
            if comp_func(queries[query][hit], threshold):
                filtered_queries[query][hit] = True
    return filtered_queries


def get_model_hits(file_path):
    queries = dict()
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:

            line = line.strip()
            line = line.split(" ")
            query = line[0]
            hit_name = line[1]
            distance = float(line[2])

            if query not in queries:
                queries[query] = dict()

            queries[query][hit_name] = distance
    return queries


def number_of_hits(queries):
    num = 0
    for key in queries:
        num += len(queries[key])
    return num


# load hits from the hmmer file.
def get_hmmer_hits(file_path):
    queries = dict()
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            line = re.split(" +", line)
            target = line[0]
            query = line[2]
            e_value = float(line[4])
            score = float(line[6])

            if query not in queries:
                queries[query] = dict()

            queries[query][target] = (e_value, score)

        return queries


def compute_matches(our_hits, hmmer_hits):
    found = []
    not_found = []
    for query in hmmer_hits:
        hmmer_query_hits = hmmer_hits[query]
        if query in our_hits:
            for hit in hmmer_query_hits.keys():
                if hit in our_hits[query]:
                    found.append(hmmer_query_hits[hit][1])
                else:
                    not_found.append(hmmer_query_hits[hit][1])
        else:
            for hit, e_value in hmmer_query_hits.items():
                not_found.append(e_value[1])

    return np.asarray(found), np.asarray(not_found)


def compute_recall(our_hits, hmmer_hits, distance_threshold, comp_func):
    match_count = 0
    for query in hmmer_hits:
        true_matches = hmmer_hits[query]
        if query in our_hits:
            our_matches = our_hits[query]
            init_match_count = match_count
            for match in our_matches:
                if match in true_matches:
                    if comp_func(our_matches[match], distance_threshold):
                        # count the matches for each query.
                        match_count += 1
            if init_match_count == match_count:
                logger.debug(
                    f"Our method did not have any of the same matches as hmmer."
                )
        else:
            logger.debug(
                f"Our method did not find any hits for query sequence {query} (hmmer found {len(true_matches)})."
            )

    # the denominator is the number of hmmer hits, which makes sense for recall.
    denominator = sum(list(map(lambda x: len(x), list(hmmer_hits.values()))))

    return 100 * (match_count / denominator)


class UniRefEvaluator(Evaluator):
    def __init__(
        self,
        query_file,
        target_file,
        nprobe,
        encoding_func,
        model_device,
        sample_percent,
        n_neighbors,
        hit_filename,
        select_random_aminos,
        query_percent,
        distance_threshold=100,
        normalize_embeddings=False,
        index_string=False,
        index_device="cpu",
    ):

        self.query_file = deepcopy(query_file)
        self.target_file = deepcopy(target_file)

        self.encoding_func = (
            wraps_tensor_for_sequence(model_device)
            if encoding_func is None
            else encoding_func
        )
        self.nprobe = nprobe
        self.model_device = model_device
        self.sample_percent = sample_percent
        self.normalize_embeddings = normalize_embeddings
        self.hit_filename = hit_filename
        self.select_random_aminos = select_random_aminos
        self.index_string = index_string
        self.index_device = index_device
        self.n_neighbors = n_neighbors
        self.query_percent = query_percent
        self.distance_threshold = distance_threshold

        if self.normalize_embeddings:
            logger.info("Using comparison function >= threshold for filtration.")
            self.comp_func = np.greater_equal
        else:
            logger.info("Using comparison function <= threshold for filtration.")
            self.comp_func = np.less_equal

        root = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/"
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/normal_hmmer_hits.txt")
        self.max_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

    @torch.no_grad()
    def _calc_or_load_embeddings(self, fasta_or_pickle_file, model_class):

        if os.path.splitext(fasta_or_pickle_file)[1] == ".pkl":
            names, sequences = fasta_from_file(
                fasta_file=os.path.splitext(fasta_or_pickle_file)[0] + ".fa"
            )
            names = list(map(lambda x: x.split(" ")[0], names))
        else:
            names, sequences = fasta_from_file(fasta_file=fasta_or_pickle_file)
            names = list(map(lambda x: x.split(" ")[0], names))

        pkl_file = os.path.splitext(fasta_or_pickle_file)[0] + ".pkl"

        if os.path.splitext(fasta_or_pickle_file)[1] == ".fa" and not os.path.isfile(
            pkl_file
        ):
            embeddings = []
            i = 0
            for sequence in tqdm.tqdm(sequences):
                if len(sequence) <= 10:
                    logger.debug(
                        f"Removing sequence {sequence} with length {len(sequence)}"
                    )
                    continue
                try:
                    embed = model_class(
                        self.encoding_func(sequence).unsqueeze(0)
                    ).squeeze(0)
                except KeyError:
                    logger.debug(f"keyerror: skipping {sequence}")
                embeddings.append(embed.transpose(-1, -2).to("cpu"))
                i += 1

            outf = os.path.splitext(fasta_or_pickle_file)[0] + ".pkl"
            logger.info(f"Saving embeddings to {outf}.")
            with open(outf, "wb") as dst:
                pickle.dump(embeddings, dst)
        else:
            logger.info(f"Loading embeddings from {fasta_or_pickle_file}.")
            with open(pkl_file, "rb") as src:
                embeddings = pickle.load(src)

        return names, sequences, embeddings

    def evaluate(self, model_class):

        if not os.path.isfile(self.hit_filename) or self.overwrite:
            (
                query_names,
                query_sequences,
                query_embeddings,
            ) = self._calc_or_load_embeddings(self.query_file, model_class=model_class)

            (
                target_names,
                target_sequences,
                target_embeddings,
            ) = self._calc_or_load_embeddings(self.target_file, model_class=model_class)

            self._setup_target_and_query_dbs(
                target_embeddings, query_embeddings, target_names, query_names
            )

            hits, avg_it, total_t = self.filter(
                query_embeddings, target_embeddings, query_names, target_names
            )

            with open("timings.txt", "w") as dst:
                dst.write(f"avg.time/it:{avg_it}, total_time:{total_t}")

            with open(self.hit_filename, "w") as file:
                for key in hits:
                    for entry in range(len(hits[key])):
                        query = key
                        target = hits[key][entry][0]
                        distance = hits[key][entry][1]
                        file.write(query + " " + target + " " + str(distance) + "\n")
        else:
            logger.info(f"Loading hits from {self.hit_filename}.")

        our_hits = get_model_hits(self.hit_filename)
        self._plot(our_hits, self.max_hmmer_hits, self.normal_hmmer_hits)

    def _plot(self, our_hits, max_hits, normal_hits):

        logger.info(f"Max hit num: {number_of_hits(max_hits)}")
        logger.info(f"Normal hit num: {number_of_hits(normal_hits)}")
        logger.info(f"Our hit num: {number_of_hits(our_hits)}")

        normal_recall = compute_recall(
            our_hits, normal_hits, self.distance_threshold, self.comp_func
        )
        max_recall = compute_recall(
            our_hits, max_hits, self.distance_threshold, self.comp_func
        )

        logger.info("Using 2000*30000 as the total number of possible matches.")
        our_filtered_hits = filter_hits(
            our_hits, self.distance_threshold, self.comp_func
        )
        filtration = 1.0 - (number_of_hits(our_filtered_hits) / (2000 * 30000))
        logger.info(
            f"Our method filtered {filtration * 100:.3f}% of possible query/target pairs."
        )
        logger.info(f"Our method got {normal_recall:.3f}% of hmmer normal hits.")
        logger.info(f"Our method got {max_recall:.3f}% of hmmer max hits.")
        # now, get the hmmer hits at different e-values
        # and compute whether or not we got them.
        found, not_found = compute_matches(our_filtered_hits, normal_hits)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(13, 13))
        ax[0, 0].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(0, 40),
            label=["found", "not_found"],
        )
        ax[0, 1].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(10, 40),
            label=["found", "not_found"],
        )
        normal_recall_over_10 = len(found[found > 10]) / (
            len(found[found > 10]) + len(not_found[not_found > 10])
        )
        ax[0, 0].set_title(
            f"recall: {normal_recall:.3f}%\n" f"filtration: {filtration * 100:.5f}%"
        )
        ax[0, 1].set_title(
            f"recall: {100 * normal_recall_over_10:.3f}%\n"
            f"filtration: {filtration * 100:.5f}%"
        )

        found, not_found = compute_matches(our_filtered_hits, max_hits)

        ax[1, 0].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(0, 40),
            label=["found", "not_found"],
        )

        ax[1, 1].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(10, 40),
            label=["found", "not_found"],
        )
        ax[1, 0].set_title(
            f"recall: {max_recall:.3f}%\n" f"filtration: {filtration * 100:.5f}%"
        )
        max_recall_over_10 = len(found[found > 10]) / (
            len(found[found > 10]) + len(not_found[not_found > 10])
        )
        ax[1, 1].set_title(
            f"recall: {100 * max_recall_over_10:.3f}%\n"
            f"filtration: {filtration * 100:.5f}%"
        )

        plt.suptitle(
            f"distance threshold: {self.distance_threshold}"
            f"\nhit file: {os.path.basename(os.path.splitext(self.hit_filename)[0])}",
            fontsize=20,
        )

        plt.savefig(
            f"{os.path.splitext(self.hit_filename)[0]}_{self.distance_threshold}.png",
            bbox_inches="tight",
        )

        # plt.savefig(f"result_figure.png", bbox_inches="tight")

        return 0

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):
        raise NotImplementedError()

    def search(self, query_embedding):
        """
        :param query_embedding: seq_lenxembedding dimension query to search against the target database.
        :type query_embedding: torch.Tensor()
        :return:
        :rtype:
        """
        raise NotImplementedError()

    @torch.no_grad()
    def filter(
        self,
        queries,
        targets,
        query_names,
        target_names,
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

            logger.info(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        return qdict, loop_time / i, loop_time


class UniRefFaissEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_target_and_query_dbs(self):
        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        unrolled_names = []

        if self.select_random_aminos:
            for i, (length, name, target) in enumerate(
                zip(lengths, target_names, targets)
            ):
                # sample N% of aminos from each sequence randomly.
                n_sample = length * self.sample_percent
                sampled_idx = torch.randperm(length)[: int(n_sample)]
                logger.debug(
                    f"random sampling: sampled_index length: {len(sampled_idx)}. original length: {length}"
                )

                if len(sampled_idx) == 0:
                    sampled_idx = [0]

                sampled_aminos = torch.cat(
                    [target[j].unsqueeze(0) for j in sampled_idx], dim=0
                )
                unrolled_names.extend([name] * len(sampled_aminos))
                unrolled_targets.append(sampled_aminos)

            unrolled_targets = torch.cat(unrolled_targets, dim=0)
        else:
            for i, (length, name, target) in enumerate(
                zip(lengths, target_names, targets)
            ):
                n_sample = length * self.sample_percent
                # sample every N amino.
                sampled_idx = torch.arange(length)[:: int(length / n_sample)]
                logger.debug(
                    f"sampled_index length: {len(sampled_idx)}. original length: {length}"
                )

                if len(sampled_idx) == 0:
                    sampled_idx = [0]

                sampled_aminos = torch.cat(
                    [target[j].unsqueeze(0) for j in sampled_idx], dim=0
                )
                unrolled_names.extend([name] * len(sampled_aminos))
                unrolled_targets.append(sampled_aminos)

            unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")
        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            nprobe=self.nprobe,
            device=self.index_device,
        )

        logger.info("Adding targets to index.")

        index.add(unrolled_targets)

    def search(self, query_embedding):

        filtered_list = []

        D, I = search_index_device_aware(
            self.index,
            query_embedding.contiguous(),
            self.index_device,
            n_neighbors=self.n_neighbors,
        )
        cnt = 2

        while torch.all(self.comp_func(D, threshold)):
            logger.debug(
                "having to increase size of search for"
                f" each query AA to {self.n_neighbors * cnt}."
            )
            if (self.n_neighbors * cnt) >= 2048 and "cuda" in self.index_device:
                logger.debug(
                    "Breaking loop, can't search for more than 2048 neighbors on GPU."
                )
                break

            D, I = search_index_device_aware(
                self.index,
                query_embedding.contiguous(),
                self.index_device,
                n_neighbors=self.n_neighbors * cnt,
            )
            cnt += 1

        for distance, idx in zip(D.ravel(), I.ravel()):
            if self.comp_func(distance, threshold):
                filtered_list.append((unrolled_names[int(idx)], distance.item()))
        return filtered_list

    def filter(
        self,
        queries,
        targets,
        query_names,
        target_names,
        threshold,
        start=0,
        end=torch.inf,
    ):
        qdict = dict()
        # construct index.

        num_queries = min(end, len(queries))
        # for query in queries
        t_tot = 0
        t_begin = time.time()
        logger.info("Beginning search.")

        logger.info(f"Selecting {self.query_percent * 100}% of query aminos.")

        for i in range(start, num_queries):
            loop_begin = time.time()
            logger.debug(f"{i / (num_queries - start):.3f}")

            filtered_list = []

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


class UniRefBruteForceEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search(self, query_embedding):
        filtered_list = []
        val = torch.min(torch.cdist(query_embedding, target_embedding))
        if val <= threshold:
            filtered_list.append((target_names[j], val.item(), j))
        return filtered_list

    def _setup_target_and_query_dbs(self):
        pass


class UniRefAlignmentEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_target_and_query_dbs(self):
        pass

    def search(self, query_embedding, target_embedding):
        filtered_list = []
        dot_products = -torch.cdist(qval, tval).to("cpu").numpy()
        ali_score = compute_ali_score(dot_products) / min(qval.shape[0], tval.shape[0])
        if ali_score <= threshold:
            filtered_list.append((target_names[j], ali_score, j))
        return filtered_list


class UniRefScannEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search(self, query_embedding):
        filtered_list = []
        I, D = searcher.search_batched(query_embedding)
        for distance, idx in zip(D.ravel(), I.ravel()):
            if distance <= threshold:
                filtered_list.append((unrolled_names[int(idx)], distance.item()))
        return filtered_list

    def _setup_target_and_query_dbs(self):

        lengths = list(map(lambda s: s.shape[0], targets))

        if self.select_random_aminos:
            unrolled_targets = torch.cat(targets, dim=0)
            unrolled_names = []
            # create a new device for getting the name of the
            # target sequence (this could actually be _way_ easier and faster; but it's fine for now.
            for i, (length, name) in enumerate(zip(lengths, target_names)):
                unrolled_names.extend([name] * length)

            assert len(unrolled_names) > 0
        else:
            # ko
            # do something clever.
            unrolled_targets = []
            unrolled_names = []
            logger.info(f"Original DB size: {sum(lengths)}")
            for i, (length, name, target) in enumerate(
                zip(lengths, target_names, targets)
            ):
                # sample N% of aminos from each sequence
                n_sample = length * self.sample_percent
                # sample every N amino.
                sampled_idx = torch.arange(length)[:: int(length / n_sample)]
                logger.debug(
                    f"sampled_index length: {len(sampled_idx)}. original length: {length}"
                )

                if len(sampled_idx) == 0:
                    sampled_idx = [0]

                sampled_aminos = torch.cat(
                    [target[j].unsqueeze(0) for j in sampled_idx], dim=0
                )
                unrolled_names.extend([name] * len(sampled_aminos))
                unrolled_targets.append(sampled_aminos)

            unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        self.searcher = (
            scann.scann_ops_pybind.builder(unrolled_targets, 7, "squared_l2")
            .tree(num_leaves=2000, num_leaves_to_search=3, training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(20)
            .build()
        )


class UniRefVAEEvaluator(UniRefEvaluator):
    def __init__(self, seq_len, overwrite, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.overwrite = overwrite

    def _setup_target_and_query_dbs(self, targets, queries, target_names, query_names):

        lengths = list(map(lambda s: s.shape[0], targets))
        logger.info(f"Original DB size: {sum(lengths)}")
        unrolled_targets = []
        self.unrolled_names = []

        if self.select_random_aminos:
            for i, (length, name, target) in enumerate(
                zip(lengths, target_names, targets)
            ):
                # sample N% of aminos from each sequence randomly.
                n_sample = length * self.sample_percent
                sampled_idx = torch.randperm(length)[: int(n_sample)]
                logger.debug(
                    f"random sampling: sampled_index length: {len(sampled_idx)}. original length: {length}"
                )

                if len(sampled_idx) == 0:
                    sampled_idx = [0]

                sampled_aminos = torch.cat(
                    [target[j].unsqueeze(0) for j in sampled_idx], dim=0
                )
                self.unrolled_names.extend([name] * len(sampled_aminos))
                unrolled_targets.append(sampled_aminos)

            unrolled_targets = torch.cat(unrolled_targets, dim=0)
        else:
            for i, (length, name, target) in enumerate(
                zip(lengths, target_names, targets)
            ):
                n_sample = length * self.sample_percent
                # sample every N amino.
                sampled_idx = torch.arange(length)[:: int(length / n_sample)]
                logger.debug(
                    f"sampled_index length: {len(sampled_idx)}. original length: {length}"
                )

                if len(sampled_idx) == 0:
                    sampled_idx = [0]

                sampled_aminos = torch.cat(
                    [target[j].unsqueeze(0) for j in sampled_idx], dim=0
                )
                self.unrolled_names.extend([name] * len(sampled_aminos))
                unrolled_targets.append(sampled_aminos)

            unrolled_targets = torch.cat(unrolled_targets, dim=0)

        logger.info(f"Number of aminos in target DB: {unrolled_targets.shape[0]}")

        if self.normalize_embeddings:
            unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

        self.index = create_faiss_index(
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
        # I think that we're hitting _every_ single family. Not sure though.
        # AAAUGH.

        if D.numel() > 0:
            logger.debug(f"min: {torch.min(D)}, max: {torch.max(D)}")
            logger.debug(
                f"Number of matches at a distance threshold of {self.distance_threshold}: {unique_names.shape}."
                f" Because of faiss limitations, actual distance threshold achieved is {torch.min(D)}"
            )

        for distance, name in zip(unique_distances.ravel(), unique_names.ravel()):
            filtered_list.append((name, distance.item()))

        return filtered_list

    @torch.no_grad()
    def _calc_or_load_embeddings(self, fasta_or_pickle_file, model_class):

        if os.path.splitext(fasta_or_pickle_file)[1] == ".pkl":
            names, sequences = fasta_from_file(
                fasta_file=os.path.splitext(fasta_or_pickle_file)[0] + ".fa"
            )
            names = list(map(lambda x: x.split(" ")[0], names))
        else:
            names, sequences = fasta_from_file(fasta_file=fasta_or_pickle_file)
            names = list(map(lambda x: x.split(" ")[0], names))

        outf = (
            os.path.splitext(fasta_or_pickle_file)[0]
            + f"{model_class.apply_cnn_loss}_{model_class.initial_seq_len}_{model_class.downsample_steps}.pkl"
        )

        remove_name_idx = []

        if (
            os.path.splitext(fasta_or_pickle_file)[1] == ".fa"
            and not os.path.isfile(outf)
        ) or self.overwrite:
            embeddings = []
            i = 0
            for j, sequence in enumerate(tqdm.tqdm(sequences)):
                if len(sequence) < model_class.initial_seq_len:
                    logger.debug(
                        f"Removing sequence {sequence} with length {len(sequence)}"
                    )
                    remove_name_idx.append(j)
                    continue
                try:

                    slices = []
                    encoded = self.encoding_func(sequence).unsqueeze(0)
                    # ok, this worked! woo-hoo!
                    for i in range(
                        0,
                        len(sequence) - model_class.initial_seq_len,
                        model_class.initial_seq_len,
                    ):
                        embed, _ = model_class(
                            encoded[:, :, i : i + model_class.initial_seq_len]
                        )
                        slices.append(embed.squeeze().T)
                    embed, _ = model_class(
                        encoded[:, :, -model_class.initial_seq_len :]
                    )
                    slices.append(embed.squeeze().T)
                    embed = torch.cat(slices, dim=0)
                    embeddings.append(embed)

                except KeyError:
                    logger.debug(f"keyerror: skipping {sequence}")
                i += 1

            logger.info(f"Saving embeddings to {outf}.")
            with open(outf, "wb") as dst:
                pickle.dump(embeddings, dst)
        else:
            logger.info(f"Loading embeddings from {fasta_or_pickle_file}.")
            with open(outf, "rb") as src:
                embeddings = pickle.load(src)

        _names = []
        for j, sequence in enumerate(sequences):
            if len(sequence) < model_class.initial_seq_len:
                remove_name_idx.append(j)

        for i, name in enumerate(names):
            if i not in remove_name_idx:
                _names.append(name)

        names = _names

        assert len(names) == len(embeddings)

        return names, sequences, embeddings
