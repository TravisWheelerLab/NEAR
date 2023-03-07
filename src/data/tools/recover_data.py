"""Functions to add the full sequences into the alignment data
Which we will then use for padding"""
import argparse
import itertools
import os

from src.data.hmmerhits import FastaFile

# TRAINING


def write_to_train_paths(task_id):
    """write the full query into the alignment file"""
    train_path = "/xdisk/twheeler/daphnedemekas/train-alignments"

    targets = list(range(45))
    queries = list(range(2, 5))
    target_queries = list(itertools.product(targets, queries))

    target = target_queries[int(task_id) - 1][0]
    query = target_queries[int(task_id) - 1][1]

    queryfile = f"uniref/split_subset/queries/queries_{query}.fa"
    queryfasta = FastaFile(queryfile)
    print(query)

    # all_hits = {}
    querysequences = queryfasta.data
    print(target)
    targetsequences = {}

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{target}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    for alignment_file in os.listdir(f"{train_path}/{query}/{target}"):
        with open(f"{train_path}/{query}/{target}/{alignment_file}", "r") as file:
            lines = file.readlines()
            if len(lines) > 3:
                file.close()
                continue
            query_and_target = lines[0]
            query = query_and_target.split()[0].strip(">")
            target = query_and_target.split()[-1]
            query_sequence = querysequences[query]
            target_sequence = targetsequences[target]

            with open(f"{train_path}/{query}/{target}/{alignment_file}", "a") as writefile:
                writefile.write("\n" + query_sequence + "\n")
                writefile.write(target_sequence + "\n")


# EVAL
def write_to_eval_paths(task_id):
    """write the full query into the eval alignment files"""
    eval_path = "/xdisk/twheeler/daphnedemekas/eval-alignments"

    target = task_id
    query = 0

    queryfile = f"uniref/split_subset/queries/queries_{query}.fa"
    queryfasta = FastaFile(queryfile)
    print(query)

    querysequences = queryfasta.data

    print(target)
    targetsequences = {}

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{target}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    for alignment_file in os.listdir(f"{eval_path}/{query}/{target}"):
        with open(f"{eval_path}/{query}/{target}/{alignment_file}", "r") as file:
            lines = file.readlines()
            query_and_target = lines[0]
            query = query_and_target.split()[0].strip(">")
            target = query_and_target.split()[-1]
            query_sequence = querysequences[query]
            target_sequence = targetsequences[target]

            with open(f"{eval_path}/{query}/{target}/{alignment_file}", "a") as writefile:
                writefile.write("\n" + query_sequence + "\n")
                writefile.write(target_sequence + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

# write_to_eval_paths(args.task_id)
write_to_train_paths(args.task_id)
