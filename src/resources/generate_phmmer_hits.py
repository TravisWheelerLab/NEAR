#!/home/u4/colligan//miniconda3/envs/faiss/bin/python
import os
import pdb
import subprocess
from argparse import ArgumentParser
from collections import defaultdict
from subprocess import check_output

import numpy as np
import tqdm


# get hits
# dump to text file formatted as query_name: target hits
def run(phmmer_file_subset):
    print(phmmer_file_subset)

    query_to_targets = defaultdict(list)
    with open(phmmer_file_subset, "r") as src:
        for line in src.readlines():
            if line[0] != "#":
                line = line.split()
                target = line[0]
                query = line[2]
                e_value = line[4]
                query_to_targets[query].append((target, float(e_value)))

    # num_queries = len(query_to_targets)
    # target_names = set()
    # 30k names in this set
    # construct the alignment queries
    # for ls in query_to_targets.values():
    #     for elem in ls:
    #         target_names.add(elem[0])

    for query, list_of_hits in tqdm.tqdm(query_to_targets.items()):
        _, query_seq = (
            check_output(f"esl-sfetch Q_benchmark2k30k.fa {query}".split())
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        query_out_file = f"query_{os.path.splitext(phmmer_file_subset)[1]}.fa"
        with open(f"{query_out_file}", "w") as dst:
            dst.write(f">{query}\n{query_seq}\n")

        for hit, e_value in list_of_hits:
            target_seq = check_output(
                f"esl-sfetch T_benchmark2k30k.fa {hit}".split()
            ).decode("utf-8")
            target_seq = target_seq[target_seq.find("\n") + 1 :]

            target_out_file = f"targets_{os.path.splitext(os.path.basename(phmmer_file_subset))[1]}.fa"

            with open(f"{target_out_file}", "w") as dst:
                dst.write(f">{query}\n{query_seq}\n")
                dst.write(f">{hit}\n{target_seq}\n")
                # save in query_target hit format.
            command = f"phmmer --max --cpu 1 -A /xdisk/twheeler/colligan/data/prefilter/alignments/{query}_{hit}.ali {query_out_file} {target_out_file}"
            alignment_output = check_output(command.split())


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("file")
    run(ap.parse_args().file)
