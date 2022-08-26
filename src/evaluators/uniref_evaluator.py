import pdb
import re
import time
from collections import defaultdict

import torch

from src.evaluators import Evaluator
from src.utils import create_faiss_index, fasta_from_file, search_index_device_aware


def get_model_hits(file_path):
    queries = dict()
    with open(file_path, "r") as file:
        lines = file.readlines()
        # print("lines: ",len(lines))
        for line in lines:
            line = line.strip()
            line = line.split(" ")
            query = line[0]
            key = line[1]
            value = float(line[2])

            # print(query, key)

            if query not in queries:
                queries[query] = dict()
            queries[query][key] = value
    return queries


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


class UniRefEvaluator(Evaluator):
    def __init__(
        self,
        query_file,
        target_file,
        encoding_func,
        use_faiss,
        n_neighbors,
        normalize_embeddings=False,
        quantize_index=False,
        index_device="cpu",
    ):

        self.query_names, self.queries = fasta_from_file(fasta_file=query_file)
        self.target_names, self.targets = fasta_from_file(fasta_file=target_file)
        # a function to encode a sequence to a tensor from its text representation
        self.encoding_func = encoding_func
        self.normalize_embeddings = normalize_embeddings
        self.use_faiss = use_faiss
        self.quantize_index = quantize_index
        self.index_device = index_device
        self.n_neighbors = n_neighbors

        root = "/home/u4/colligan/data/prefilter/uniref_benchmark/"
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/normal_hmmer_hits.txt")
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

    def _calc_embeddings(self, model_class, sequences):
        embeddings = []
        i = 0

        for sequence in sequences:
            embed = model_class(self.encoding_func(sequence).unsqueeze(0)).squeeze(0)
            embeddings.append(embed.transpose(-1, -2))
            print(f"{i}, {len(sequences)} {i / len(sequences):.3f}", end="\r")
            i += 1

        return embeddings

    @torch.no_grad()
    def evaluate(self, model_class):
        """Must have already loaded model weights."""
        # smash the queries and targets against each other
        # why did daniel sort the sequences by length?

        query_embeddings = self._calc_embeddings(model_class, self.queries)
        target_embeddings = self._calc_embeddings(model_class, self.targets)

        print("Computing hits.")

        if self.use_faiss:
            hits = self.filter_with_faiss(
                query_embeddings,
                target_embeddings,
                self.query_names,
                self.target_names,
                threshold=100.0,
            )
        else:
            hits = self.filter_exhaustive(
                query_embeddings,
                target_embeddings,
                self.query_names,
                self.target_names,
                threshold=100.0,
            )

        pdb.set_trace()
        exit()
        with open("./hits.txt", "w") as file:
            for key in hits:
                for entry in range(len(hits[key])):
                    query = key
                    target = hits[key][entry][0]
                    distance = float(hits[key][entry][1])
                    file.write(query + " " + target + " " + str(distance) + "\n")

        our_hits = get_model_hits("./hits.txt")
        return 0

    def filter_with_faiss(
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
        lengths = map(lambda s: s.shape[0], targets)
        unrolled_targets = torch.cat(targets, dim=0)
        unrolled_names = []
        # create a new device for getting the name of the
        # target sequence (this could actually be _way_ easier and faster; but it's fine for now.
        for i, (length, name) in enumerate(zip(lengths, target_names)):
            unrolled_names.extend([name] * length)

        index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            quantize=self.quantize_index,
            device=self.index_device,
        )

        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0
        for i in range(start, num_queries):
            begin = time.time()
            print(f"{i / (num_queries - start):.3f}", end="\r")

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
                print(
                    "having to increase size of search for"
                    f" each query AA to {50*cnt}."
                )

                D, I = index.search(qval.contiguous(), k=50 * cnt)
                cnt += 1

            for distance, idx in zip(D.ravel(), I.ravel()):
                if distance <= threshold:
                    filtered_list.append((unrolled_names[int(idx)], distance))

            qdict[query_names[i]] = filtered_list
            time_taken = time.time() - begin
            t_tot += time_taken

            print(f"time/it: {time_taken}, avg time/it: {t_tot / (i+1)}")

        return qdict

    def filter_exhaustive(
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

        for i in range(start, num_queries):
            begin = time.time()

            print(f"{i / (num_queries - start):.3f}", end="\r")

            filtered_list = []

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

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
                    tval = torch.nn.functional.normalize(targets[j], dim=-1)
                else:
                    tval = targets[j]

                # basically, you get the minimum distance between the query embedding
                # and the target embedding.
                # I wonder which one is faster: index with queries or index with targets.
                #
                val = torch.min(torch.cdist(qval, tval))

                if val <= threshold:
                    filtered_list.append((target_names[j], val, j))

            qdict[query_names[i]] = filtered_list

            time_taken = time.time() - begin
            t_tot += time_taken

            print(f"time/it: {time_taken}, avg time/it: {t_tot / (i+1)}")

        return qdict
