import pdb
import re

import torch

from src.evaluators import Evaluator
from src.utils import fasta_from_file


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
        self, query_file, target_file, encoding_func, normalize_embeddings=False
    ):

        self.query_names, self.queries = fasta_from_file(fasta_file=query_file)
        self.target_names, self.targets = fasta_from_file(fasta_file=target_file)
        # a function to encode a sequence to a tensor from its text representation
        self.encoding_func = encoding_func
        self.normalize_embeddings = normalize_embeddings

        root = "/home/u4/colligan/data/prefilter/uniref_benchmark/"
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/normal_hmmer_hits.txt")
        self.normal_hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")

    def _calc_embeddings(self, model_class, sequences):

        embeddings = []
        for sequence in sequences:
            embed = (
                model_class(self.encoding_func(sequence).unsqueeze(0))
                .squeeze(0)
                .to("cuda")
            )
            embeddings.append(embed.transpose(-1, -2))

        return embeddings

    def evaluate(self, model_class):
        """Must have already loaded model weights."""
        # smash the queries and targets against each other
        # why did daniel sort the sequences by length?
        query_embeddings = self._calc_embeddings(model_class, self.query_names)
        target_embeddings = self._calc_embeddings(model_class, self.target_names)
        hits = self.filter(
            query_embeddings,
            target_embeddings,
            self.query_names,
            self.target_names,
            threshold=100.0,
        )

        return 0

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

        for i in range(start, num_queries):
            print(f"{i / (num_queries - start):.3f}", end="\r")

            filtered_list = []

            if self.normalize_embeddings:
                qval = torch.nn.functional.normalize(queries[i], dim=-1)
            else:
                qval = queries[i]

            for j in range(len(targets)):

                if self.normalize_embeddings:
                    tval = torch.nn.functional.normalize(targets[j], dim=-1)
                else:
                    tval = targets[j]

                val = torch.min(torch.cdist(qval, tval))

                if val <= threshold:
                    filtered_list.append((target_names[j], val, j))

            qdict[query_names[i]] = filtered_list

        return qdict
