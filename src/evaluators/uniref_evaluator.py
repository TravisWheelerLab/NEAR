import logging
import os
import pdb
import pickle
import re
import time
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numba
import numpy as np
import scann
import torch
import tqdm

from src.evaluators import Evaluator
from src.utils import create_faiss_index, fasta_from_file, search_index_device_aware

logger = logging.getLogger("evaluate")

amino_n_to_a = [c for c in "ARNDCQEGHILKMFPSTWYVBZXJ*U"]
amino_a_to_n = {c: i for i, c in enumerate("ARNDCQEGHILKMFPSTWYVBZXJ*U")}
amino_frequencies = torch.tensor(
    [
        0.074,
        0.042,
        0.044,
        0.059,
        0.033,
        0.058,
        0.037,
        0.074,
        0.029,
        0.038,
        0.076,
        0.072,
        0.018,
        0.040,
        0.050,
        0.081,
        0.062,
        0.013,
        0.033,
        0.068,
    ]
)

amino_n_to_v = torch.zeros(len(amino_n_to_a), 20)
for i in range(20):
    amino_n_to_v[i, i] = 1.0

amino_n_to_v[amino_a_to_n["B"], amino_a_to_n["D"]] = 0.5
amino_n_to_v[amino_a_to_n["B"], amino_a_to_n["N"]] = 0.5

amino_n_to_v[amino_a_to_n["Z"], amino_a_to_n["Q"]] = 0.5
amino_n_to_v[amino_a_to_n["Z"], amino_a_to_n["E"]] = 0.5

amino_n_to_v[amino_a_to_n["J"], amino_a_to_n["I"]] = 0.5
amino_n_to_v[amino_a_to_n["J"], amino_a_to_n["L"]] = 0.5

amino_n_to_v[amino_a_to_n["X"]] = amino_frequencies
amino_n_to_v[amino_a_to_n["*"]] = amino_frequencies
amino_n_to_v[amino_a_to_n["U"]] = amino_frequencies

amino_a_to_v = {c: amino_n_to_v[i] for i, c in enumerate("ARNDCQEGHILKMFPSTWYVBZXJ*U")}


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


def wraps_tensor_for_sequence(device):
    def tensor_for_sequence(sequence):
        data = torch.zeros(20, len(sequence))
        for i, c in enumerate(sequence):
            data[:, i] = amino_a_to_v[c]
        return data.to(device)

    return tensor_for_sequence


def filtered_hits(queries, threshold):
    filtered_queries = dict()
    for query in queries:
        filtered_queries[query] = dict()
        for hit in queries[query]:
            if queries[query][hit] <= threshold:
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


def union_hits(query1, query2):
    queries = dict()

    # where do the two dictionaries overlap?
    for qkey in query1:
        queries[qkey] = dict()
        if qkey in query2:
            for hit in query1[qkey]:
                if hit in query2[qkey]:
                    queries[qkey][hit] = True

    return queries


def union_queries(primary, secondary):
    new_secondary = dict()
    # create a new dictionary with only the queries
    # that appear in primary. This _removes_
    # hits from hmmer matches to only contain the ones
    # that are present in our hits
    for qkey in primary:
        if qkey in secondary:
            new_secondary[qkey] = secondary[qkey]
        else:
            secondary[qkey] = dict()
    return new_secondary


def scores(hmmer, other):
    found = []
    not_found = []
    for qkey in other:
        for tkey in hmmer[qkey]:
            if tkey in other[qkey]:
                found.append(hmmer[qkey][tkey][1])
            else:
                not_found.append(hmmer[qkey][tkey][1])
    return found, not_found


def all_our_scores(hits):
    values = []
    for qkey in hits:
        for tkey in hits[qkey]:
            values.append(hits[qkey][tkey])

    return torch.tensor(values)


class UniRefEvaluator(Evaluator):
    def __init__(
        self,
        query_file,
        target_file,
        nprobe,
        encoding_func,
        filter_value,
        model_device,
        sample_percent,
        n_neighbors,
        hit_filename,
        select_random_aminos,
        distance_threshold=100,
        normalize_embeddings=False,
        index_string=False,
        index_device="cpu",
    ):

        self.query_file = deepcopy(query_file)
        self.target_file = deepcopy(target_file)

        self.encoding_func = wraps_tensor_for_sequence(model_device)
        self.nprobe = nprobe
        self.filter_value = filter_value
        self.model_device = model_device
        self.sample_percent = sample_percent
        self.normalize_embeddings = normalize_embeddings
        self.hit_filename = hit_filename
        self.select_random_aminos = select_random_aminos
        self.index_string = index_string
        self.index_device = index_device
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold

        root = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/"
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/normal_hmmer_hits.txt")
        self.max_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

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
        if not os.path.isfile(self.hit_filename):
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

            hits, avg_it, total_t = self.filter(
                query_embeddings,
                target_embeddings,
                query_names,
                target_names,
                threshold=self.distance_threshold,
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

        logger.info(number_of_hits(max_hits))
        logger.info(number_of_hits(normal_hits))
        logger.info(number_of_hits(our_hits))

        values = all_our_scores(our_hits)
        values, _ = torch.sort(values)

        small_max = union_queries(our_hits, max_hits)
        small_normal = union_queries(our_hits, normal_hits)
        our_filter = filtered_hits(our_hits, self.filter_value)

        logger.info("Using 2000*30000 as the total number of possible matches.")
        # filtration = 1.0 - (number_of_hits(our_filter) / number_of_hits(our_hits))
        filtration = 1.0 - (number_of_hits(our_filter) / 2000 * 30000)

        logger.info(f"Filtered {filtration} at a distance of {self.filter_value}")
        logger.info(
            f"max: {number_of_hits(small_max)}, {number_of_hits(union_hits(our_filter, small_max))}"
        )

        logger.info(
            f"normal: {number_of_hits(small_normal)}, {number_of_hits(union_hits(our_filter, small_normal))}"
        )

        our_filter = filtered_hits(our_hits, self.filter_value)

        small_normal = union_queries(our_hits, normal_hits)

        filtration = 1.0 - (number_of_hits(our_filter) / (2000 * 30000))

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(15.0, 15)
        plt.suptitle(
            f"filter threshold: {self.filter_value} \n"
            f"hit file: {os.path.basename(os.path.splitext(self.hit_filename)[0])}",
            fontsize=20,
        )

        found, not_found = scores(max_hits, our_filter)

        recall = len(found) / (len(found) + len(not_found))

        found, _ = torch.sort(torch.tensor(found))
        not_found, _ = torch.sort(torch.tensor(not_found))

        axs[0, 0].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(0, 40),
            label=["found", "not_found"],
        )
        axs[0, 0].text(25, 2000, f"filtration: {filtration:.3f}\n recall: {recall:.3f}")

        axs[0, 0].set_title("hmmsearch max vs our filter")

        axs[0, 0].legend()

        axs[0, 1].hist(
            [found[found > 10], not_found[not_found > 10]],
            stacked=True,
            bins=10,
            range=(10, 40),
            label=["found", "not_found"],
        )
        axs[0, 1].legend()
        recall = len(found[found > 10]) / (
            len(found[found > 10]) + len(not_found[not_found > 10])
        )
        axs[0, 1].text(25, 100, f"filtration: {filtration:.3f}\n recall: {recall:.3f}")

        our_filter = filtered_hits(our_hits, self.filter_value)
        found, not_found = scores(normal_hits, our_filter)
        recall = len(found) / (len(found) + len(not_found))
        found, _ = torch.sort(torch.tensor(found))
        not_found, _ = torch.sort(torch.tensor(not_found))

        axs[1, 0].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(0, 40),
            label=["found", "not_found"],
        )
        axs[1, 0].text(25, 2000, f"filtration: {filtration:.3f}\n recall: {recall:.3f}")

        axs[1, 0].set_title("normal hmmsearch vs our filter")

        axs[1, 0].legend()

        axs[1, 1].hist(
            [found[found > 10], not_found[not_found > 10]],
            stacked=True,
            bins=10,
            range=(10, 40),
            label=["found", "not_found"],
        )
        axs[1, 1].legend()

        recall = len(found[found > 10]) / (
            len(found[found > 10]) + len(not_found[not_found > 10])
        )
        axs[1, 1].text(25, 2000, f"filtration: {filtration:.3f}\n recall: {recall:.3f}")

        axs[0, 0].set_ylabel("Sequences found")
        axs[0, 1].set_ylabel("Sequences found")
        axs[1, 0].set_ylabel("Sequences found")
        axs[1, 1].set_ylabel("Sequences found")

        axs[0, 0].set_xlabel("HMMER score")
        axs[0, 1].set_xlabel("HMMER score")
        axs[1, 0].set_xlabel("HMMER score")
        axs[1, 1].set_xlabel("HMMER score")

        plt.savefig(
            f"{os.path.splitext(self.hit_filename)[0]}_{self.filter_value}.png",
            bbox_inches="tight",
        )

        plt.savefig(f"result_figure.png", bbox_inches="tight")

        return 0

    def filter(self, *args, **kwargs):
        raise NotImplementedError()


class UniRefFaissEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            nprobe=self.nprobe,
            device=self.index_device,
        )

        logger.info("Adding targets to index.")

        index.add(unrolled_targets)

        logger.info("Beginning search.")

        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0
        t_begin = time.time()

        for i in range(start, num_queries):
            loop_begin = time.time()
            logger.debug(f"{i / (num_queries - start):.3f}")

            filtered_list = []

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

            D, I = search_index_device_aware(
                index,
                qval.contiguous(),
                self.index_device,
                n_neighbors=self.n_neighbors,
            )
            cnt = 2

            while torch.all(D <= threshold):
                logger.debug(
                    "having to increase size of search for"
                    f" each query AA to {50 * cnt}."
                )
                if (self.n_neighbors * cnt) >= 2048 and "cuda" in self.index_device:
                    logger.debug(
                        "Breaking loop, can't search for more than 2048 neighbors on GPU."
                    )
                    break

                D, I = search_index_device_aware(
                    index,
                    qval.contiguous(),
                    self.index_device,
                    n_neighbors=self.n_neighbors * cnt,
                )
                cnt += 1

            if self.normalize_embeddings:
                for distance, idx in zip(D.ravel(), I.ravel()):
                    if distance >= threshold:
                        filtered_list.append(
                            (unrolled_names[int(idx)], distance.item())
                        )
            else:
                for distance, idx in zip(D.ravel(), I.ravel()):
                    if distance <= threshold:
                        filtered_list.append(
                            (unrolled_names[int(idx)], distance.item())
                        )

            qdict[query_names[i]] = filtered_list
            time_taken = time.time() - loop_begin
            t_tot += time_taken

            logger.debug(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        return qdict, loop_time / i, loop_time


class UniRefBruteForceEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
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

        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0
        logger.info("Starting alignment-aware distances.")

        for i in range(start, num_queries):
            begin = time.time()

            logger.info(f"{i / (num_queries - start):.3f}")

            filtered_list = []

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(
                    queries[i].to(self.model_device), dim=-1
                )
            else:
                qval = queries[i].to(self.model_device)

            # get the minimum distance between each
            # of the sequences in the query set and the target set
            # if the val is less than or equal to the threshold,
            # then record it as a hit;
            # each query then gets a record of (target hit, distance, and index into target)
            # this can be transformed:
            # for each query in queries:
            #     range_search(index_of_targets, threshold)
            for j in range(len(targets)):

                if self.normalize_embeddings:
                    tval = torch.nn.functional.normalize(
                        targets[j].to(self.model_device), dim=-1
                    )
                else:
                    tval = targets[j].to(self.model_device)

                # basically, you get the minimum distance between the query embedding
                # and the target embedding.
                # I wonder which one is faster: index with queries or index with targets.

                val = torch.min(torch.cdist(qval, tval))

                if val <= threshold:
                    filtered_list.append((target_names[j], val.item(), j))

            qdict[query_names[i]] = filtered_list

            time_taken = time.time() - begin
            t_tot += time_taken

            logger.debug(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        return qdict, t_tot / i, t_tot


class UniRefAlignmentEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
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
        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0
        logger.info("Starting brute-force cdist.")

        for i in range(start, num_queries):
            begin = time.time()

            logger.info(f"{i / (num_queries - start):.3f}")

            filtered_list = []

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(
                    queries[i].to(self.model_device), dim=-1
                )
            else:
                qval = queries[i].to(self.model_device)

            # get the minimum distance between each
            # of the sequences in the query set and the target set
            # if the val is less than or equal to the threshold,
            # then record it as a hit;
            # each query then gets a record of (target hit, distance, and index into target)
            # this can be transformed:
            # for each query in queries:
            #     range_search(index_of_targets, threshold)
            for j in range(len(targets)):

                if self.normalize_embeddings:
                    tval = torch.nn.functional.normalize(
                        targets[j].to(self.model_device), dim=-1
                    )
                else:
                    tval = targets[j].to(self.model_device)

                # divide by the minimum length.
                dot_products = -torch.cdist(qval, tval).to("cpu").numpy()
                ali_score = compute_ali_score(dot_products) / min(
                    qval.shape[0], tval.shape[0]
                )

                logger.debug(f"Alignment score: {ali_score}")

                if ali_score <= threshold:
                    filtered_list.append((target_names[j], ali_score, j))

            qdict[query_names[i]] = filtered_list

            time_taken = time.time() - begin
            t_tot += time_taken

            logger.info(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

            return qdict, t_tot / i, t_tot


class UniRefScannEvaluator(UniRefEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
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
        num_queries = min(end, len(queries))
        # construct index.
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

        searcher = (
            scann.scann_ops_pybind.builder(unrolled_targets, 7, "squared_l2")
            .tree(num_leaves=2000, num_leaves_to_search=3, training_sample_size=250000)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(20)
            .build()
        )

        t_tot = 0
        t_begin = time.time()

        for i in range(start, num_queries):
            loop_begin = time.time()
            logger.debug(f"{i / (num_queries - start):.3f}")

            filtered_list = []

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

            I, D = searcher.search_batched(qval)

            if self.normalize_embeddings:
                for distance, idx in zip(D.ravel(), I.ravel()):
                    if distance >= threshold:
                        filtered_list.append(
                            (unrolled_names[int(idx)], distance.item())
                        )
            else:
                for distance, idx in zip(D.ravel(), I.ravel()):
                    if distance <= threshold:
                        filtered_list.append(
                            (unrolled_names[int(idx)], distance.item())
                        )

            qdict[query_names[i]] = filtered_list
            time_taken = time.time() - loop_begin
            t_tot += time_taken

            logger.debug(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        loop_time = time.time() - t_begin

        logger.info(f"Entire loop took: {loop_time}.")

        return qdict, loop_time / i, loop_time
