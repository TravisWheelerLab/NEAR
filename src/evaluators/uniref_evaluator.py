import os
import pdb
import pickle
import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from src.evaluators import Evaluator
from src.utils import create_faiss_index, fasta_from_file, search_index_device_aware

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


def wraps_tensor_for_sequence(device):
    def tensor_for_sequence(sequence):
        data = torch.zeros(20, len(sequence))
        for i, c in enumerate(sequence):
            data[:, i] = amino_a_to_v[c]
        return data.to(device)

    return tensor_for_sequence


def filtered_hits(queries, threshold):
    filtered_queries = dict()
    for key in queries:
        filtered_queries[key] = dict()
        for tkey in queries[key]:
            if queries[key][tkey] <= threshold:
                filtered_queries[key][tkey] = True
    return filtered_queries


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

            if query not in queries:
                queries[query] = dict()

            queries[query][key] = value
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
    for qkey in query1:
        queries[qkey] = dict()
        if qkey in query2:
            for tkey in query1[qkey]:
                if tkey in query2[qkey]:
                    queries[qkey][tkey] = True

    return queries


def union_queries(primary, secondary):
    new_secondary = dict()
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
        encoding_func,
        use_faiss,
        n_neighbors,
        hit_filename,
        distance_threshold=100,
        normalize_embeddings=False,
        index_string=False,
        index_device="cpu",
    ):

        self.query_names, self.queries = fasta_from_file(fasta_file=query_file)
        self.target_names, self.targets = fasta_from_file(fasta_file=target_file)
        self.query_names = list(map(lambda x: x.split(" ")[0], self.query_names))
        self.target_names = list(map(lambda x: x.split(" ")[0], self.target_names))
        # a function to encode a sequence to a tensor from its text representation
        # TODO: don't do the hardcoding below
        self.encoding_func = wraps_tensor_for_sequence("cuda")
        self.normalize_embeddings = normalize_embeddings
        self.use_faiss = use_faiss
        self.hit_filename = hit_filename
        self.index_string = index_string
        self.index_device = index_device
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold

        root = "/home/u4/colligan/data/prefilter/uniref_benchmark/"
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/normal_hmmer_hits.txt")
        self.max_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

    def _calc_embeddings(self, model_class, sequences):
        embeddings = []
        i = 0

        for sequence in sequences:
            if len(sequence) <= 10:
                print(f"Removing sequence {sequence}")
                continue
            try:
                embed = model_class(self.encoding_func(sequence).unsqueeze(0)).squeeze(
                    0
                )
            except KeyError:
                print(f"keyerror: skipping {sequence}")
            embeddings.append(embed.transpose(-1, -2).to("cpu"))

            if (i + 1) % 100000 == 0:
                with open(
                    f"/xdisk/twheeler/colligan/uniprot_sprot_{i}.pkl", "wb"
                ) as dst:
                    pickle.dump(embeddings, dst)
                    embeddings = []
            i += 1

        with open(f"/xdisk/twheeler/colligan/uniprot_sprot_{i}.pkl", "wb") as dst:
            pickle.dump(embeddings, dst)

        return embeddings

    @torch.no_grad()
    def evaluate(self, model_class):
        """Must have already loaded model weights."""
        # smash the queries and targets against each other
        # why did daniel sort the sequences by length?
        overwrite = False

        if os.path.isfile("query_embeds.pkl"):
            with open("query_embeds.pkl", "rb") as src:
                query_embeddings = pickle.load(src)
            with open("target_embeds.pkl", "rb") as src:
                target_embeddings = pickle.load(src)
        else:
            query_embeddings = self._calc_embeddings(model_class, self.queries)
            target_embeddings = self._calc_embeddings(model_class, self.targets)
            with open(
                "/xdisk/twheeler/colligan/swiss_prot_embeddings.pkl", "wb"
            ) as dst:
                pickle.dump(target_embeddings, dst)

        print("Computing hits.")

        if not os.path.isfile(self.hit_filename):
            if self.use_faiss:
                hits = self.filter_with_faiss(
                    query_embeddings,
                    target_embeddings,
                    self.query_names,
                    self.target_names,
                    threshold=self.distance_threshold,
                )
            else:
                hits = self.filter_exhaustive(
                    query_embeddings,
                    target_embeddings,
                    self.query_names,
                    self.target_names,
                    threshold=self.distance_threshold,
                )

            with open(self.hit_filename, "w") as file:
                for key in hits:
                    for entry in range(len(hits[key])):
                        query = key
                        target = hits[key][entry][0]
                        distance = hits[key][entry][1]
                        file.write(query + " " + target + " " + str(distance) + "\n")

        max_hits = get_hmmer_hits(
            "/home/u4/colligan/data/prefilter/uniref_benchmark/max_hmmer_hits.txt"
        )
        normal_hits = get_hmmer_hits(
            "/home/u4/colligan/data/prefilter/uniref_benchmark/normal_hmmer_hits.txt"
        )
        our_hits = get_model_hits(self.hit_filename)

        print(number_of_hits(max_hits))
        print(number_of_hits(normal_hits))
        print(number_of_hits(our_hits))

        values = all_our_scores(our_hits)
        values, _ = torch.sort(values)
        small_max = union_queries(our_hits, max_hits)
        small_normal = union_queries(our_hits, normal_hits)
        our_filter = filtered_hits(our_hits, 0.4)
        filtration = 1.0 - (number_of_hits(our_filter) / number_of_hits(our_hits))

        print(number_of_hits(our_filter), "=", filtration)
        print("--")
        print(
            "max:",
            number_of_hits(small_max),
            number_of_hits(union_hits(our_filter, small_max)),
        )
        print(
            "normal:",
            number_of_hits(small_normal),
            number_of_hits(union_hits(our_filter, small_normal)),
        )

        filt_value = 0.73  # 0.740 gives great results
        our_filter = filtered_hits(our_hits, filt_value)
        small_max = union_queries(our_hits, max_hits)
        small_normal = union_queries(our_hits, normal_hits)

        filtration = 1.0 - (number_of_hits(our_filter) / number_of_hits(our_hits))

        fig, axs = plt.subplots(3, 2)
        fig.set_size_inches(15.0, 15)

        # small_max = union_queries(our_filter, max_hits)
        # small_normal = union_queries(our_filter, normal_hits)

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
        # axs[0,0].hist(not_found, bins=10, color=(1,0,0,0.5),range=(0,80), label='not found')
        axs[0, 0].text(
            25, 2000, "filtration: " + str(filtration) + "\nrecall: " + str(recall)
        )

        axs[0, 0].legend()

        axs[0, 1].hist(
            [found[found > 10], not_found[not_found > 10]],
            stacked=True,
            bins=10,
            range=(10, 40),
            label=["found", "not_found"],
        )
        # axs[0,1].hist(not_found[not_found > 10], bins=10, color=(1,0,0,0.5),range=(10,80), label='not found')
        axs[0, 1].legend()
        recall = len(found[found > 10]) / (
            len(found[found > 10]) + len(not_found[not_found > 10])
        )
        axs[0, 1].text(
            25, 100, "filtration: " + str(filtration) + "\nrecall: " + str(recall)
        )

        our_filter = filtered_hits(our_hits, filt_value)

        # small_max = union_queries(our_filter, max_hits)
        # small_normal = union_queries(our_filter, normal_hits)

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
        axs[1, 0].text(
            25, 2000, "filtration: " + str(filtration) + "\nrecall: " + str(recall)
        )
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
        # axs[1,1].text(25, 100, "filtration: " + str(filtration) + "\nrecall: " + str(recall))
        found, not_found = scores(max_hits, small_normal)
        recall = len(found) / (len(found) + len(not_found))
        found, _ = torch.sort(torch.tensor(found))
        not_found, _ = torch.sort(torch.tensor(not_found))

        filtration = 0.995572
        axs[2, 0].hist(
            [found, not_found],
            stacked=True,
            bins=10,
            range=(0, 40),
            label=["found", "not_found"],
        )
        axs[2, 0].text(
            25,
            2000,
            "filtration: " + str(filtration) + "\nrecall: " + str(recall),
            label="not found",
        )
        axs[2, 0].legend()
        axs[2, 1].hist(
            [found[found > 10], not_found[not_found > 10]],
            stacked=True,
            bins=10,
            range=(10, 40),
            label=["found", "not_found"],
        )
        axs[2, 1].legend()

        recall = len(found[found > 10]) / (
            len(found[found > 10]) + len(not_found[not_found > 10])
        )
        # axs[2,1].text(25, 100, "filtration: " + str(filtration) + "\nrecall: " + str(recall))

        axs[0, 0].set_ylabel("Sequences found")
        axs[0, 1].set_ylabel("Sequences found")
        axs[1, 0].set_ylabel("Sequences found")
        axs[1, 1].set_ylabel("Sequences found")
        axs[2, 0].set_ylabel("Sequences found")
        axs[2, 1].set_ylabel("Sequences found")

        axs[0, 0].set_xlabel("HMMER score")
        axs[0, 1].set_xlabel("HMMER score")
        axs[1, 0].set_xlabel("HMMER score")
        axs[1, 1].set_xlabel("HMMER score")
        axs[2, 0].set_xlabel("HMMER score")
        axs[2, 1].set_xlabel("HMMER score")
        plt.savefig(f"/home/u4/colligan/{self.hit_filename}.png", bbox_inches="tight")

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
        lengths = list(map(lambda s: s.shape[0], targets))
        print("Number of aminos in target DB:", sum(lengths))
        unrolled_targets = torch.cat(targets, dim=0)

        unrolled_names = []
        # create a new device for getting the name of the
        # target sequence (this could actually be _way_ easier and faster; but it's fine for now.
        for i, (length, name) in enumerate(zip(lengths, target_names)):
            unrolled_names.extend([name] * length)

        assert len(unrolled_names) > 0

        index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if self.normalize_embeddings else "l2",
            index_string=self.index_string,
            device=self.index_device,
        )

        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0
        t_begin = time.time()

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
                # print(
                #     "having to increase size of search for"
                #     f" each query AA to {50 * cnt}."
                # )

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
            time_taken = time.time() - begin
            t_tot += time_taken

            print(f"time/it: {time_taken}, avg time/it: {t_tot / (i+1)}")

        print(f"Entire loop took: {time.time() - t_begin}")

        return qdict

    @torch.no_grad()
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
        print("I'm here")

        num_queries = min(end, len(queries))

        # for query in queries
        t_tot = 0

        for i in range(start, num_queries):
            begin = time.time()

            # print(f"{i / (num_queries - start):.3f}", end="\r")

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

                val = torch.min(torch.cdist(qval, tval))

                if val <= threshold:
                    filtered_list.append((target_names[j], val.item(), j))

            qdict[query_names[i]] = filtered_list

            time_taken = time.time() - begin
            t_tot += time_taken

            print(f"time/it: {time_taken}, avg time/it: {t_tot / (i + 1)}")

        return qdict
