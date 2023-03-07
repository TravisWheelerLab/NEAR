import itertools
import os

from src.data.hmmerhits import FastaFile

# TRAINING


def write_to_train_paths(id):
    task_id = int(id)

    ts = list(range(45))
    qs = list(range(2, 5))
    target_queries = list(itertools.product(ts, qs))

    t = target_queries[task_id - 1][0]
    q = target_queries[task_id - 1][1]

    queryfile = f"uniref/split_subset/queries/queries_{q}.fa"
    queryfasta = FastaFile(queryfile)
    print(q)

    # all_hits = {}
    querysequences = queryfasta.data
    print(t)
    targetsequences = {}

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{t}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    for alignment_file in os.listdir(f"/xdisk/twheeler/daphnedemekas/train-alignments/{q}/{t}"):
        file = open(f"/xdisk/twheeler/daphnedemekas/train-alignments/{q}/{t}/{alignment_file}", "r")
        lines = file.readlines()
        if len(lines) > 3:
            file.close()
            continue
        query_and_target = lines[0]
        query = query_and_target.split()[0].strip(">")
        target = query_and_target.split()[-1]
        query_sequence = querysequences[query]
        target_sequence = targetsequences[target]

        writefile = open(
            f"/xdisk/twheeler/daphnedemekas/train-alignments/{q}/{t}/{alignment_file}", "a"
        )
        writefile.write("\n" + query_sequence + "\n")
        writefile.write(target_sequence + "\n")
        writefile.close()
        file.close()


# raise

# EVAL
def write_to_eval_paths(id):
    alignment_file_paths = "/xdisk/twheeler/daphnedemekas/train_paths2.txt"

    t = id
    q = 0

    queryfile = f"uniref/split_subset/queries/queries_{q}.fa"
    queryfasta = FastaFile(queryfile)
    print(q)

    # all_hits = {}
    querysequences = queryfasta.data

    print(t)
    targetsequences = {}

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{t}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    for alignment_file in os.listdir(f"/xdisk/twheeler/daphnedemekas/eval-alignments/{q}/{t}"):
        file = open(f"/xdisk/twheeler/daphnedemekas/eval-alignments/{q}/{t}/{alignment_file}", "r")
        lines = file.readlines()
        query_and_target = lines[0]
        query = query_and_target.split()[0].strip(">")
        target = query_and_target.split()[-1]
        query_sequence = querysequences[query]
        target_sequence = targetsequences[target]

        writefile = open(
            f"/xdisk/twheeler/daphnedemekas/eval-alignments/{q}/{t}/{alignment_file}", "a"
        )
        writefile.write("\n" + query_sequence + "\n")
        writefile.write(target_sequence + "\n")
        writefile.close()
        file.close()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

# write_to_eval_paths(args.task_id)
write_to_train_paths(args.task_id)
