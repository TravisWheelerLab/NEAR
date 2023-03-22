"""Functions to add the full sequences into the alignment data
Which we will then use for padding"""
import argparse
import itertools
import os

from src.data.hmmerhits import FastaFile
import logging

# TRAINING
logger = logging.getLogger(__name__)


def write_to_train_paths(task_id):
    """write the full query into the alignment file"""
    train_path = "/xdisk/twheeler/daphnedemekas/train-alignments"

    targets = list(range(45))
    queries = list(range(2, 5))
    target_queries = list(itertools.product(targets, queries))

    target_num = target_queries[int(task_id) - 1][0]
    query_num = target_queries[int(task_id) - 1][1]

    queryfile = f"uniref/split_subset/queries/queries_{query_num}.fa"
    queryfasta = FastaFile(queryfile)
    print(query_num)

    # all_hits = {}
    querysequences = queryfasta.data
    print(target_num)
    targetsequences = {}

    targetfasta = FastaFile(
        f"uniref/split_subset/targets/targets_{target_num}.fa"
    )
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    for alignment_file in os.listdir(f"{train_path}/{query_num}/{target_num}"):
        with open(
            f"{train_path}/{query_num}/{target_num}/{alignment_file}", "r"
        ) as file:
            lines = file.readlines()
            if len(lines) > 3:
                file.close()
                continue
            else:
                print("writing to training data")
            query_and_target = lines[0]
            query = query_and_target.split()[0].strip(">")
            target = query_and_target.split()[-1]
            query_sequence = querysequences[query]
            target_sequence = targetsequences[target]

        with open(
            f"{train_path}/{query_num}/{target_num}/{alignment_file}", "a"
        ) as writefile:
            writefile.write("\n" + query_sequence + "\n")
            writefile.write(target_sequence + "\n")


# EVAL
def write_to_eval_paths(task_id):
    """write the full query into the eval alignment files"""
    eval_path = "/xdisk/twheeler/daphnedemekas/eval-alignments"

    target_num = task_id
    query_num = 0

    queryfile = f"uniref/split_subset/queries/queries_{query_num}.fa"
    queryfasta = FastaFile(queryfile)
    print(query_num)

    querysequences = queryfasta.data

    print(target_num)
    targetsequences = {}

    targetfasta = FastaFile(
        f"uniref/split_subset/targets/targets_{target_num}.fa"
    )
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    for alignment_file in os.listdir(f"{eval_path}/{query_num}/{target_num}"):
        with open(
            f"{eval_path}/{query_num}/{target_num}/{alignment_file}", "r"
        ) as file:
            lines = file.readlines()
            if len(lines) > 3:
                file.close()
                continue
            else:
                print("writing to evaluation data")
            query_and_target = lines[0]
            query = query_and_target.split()[0].strip(">")
            target = query_and_target.split()[-1]
            query_sequence = querysequences[query]
            target_sequence = targetsequences[target]

        with open(
            f"{eval_path}/{query_num}/{target_num}/{alignment_file}", "a"
        ) as writefile:
            writefile.write("\n" + query_sequence + "\n")
            writefile.write(target_sequence + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

write_to_train_paths(args.task_id)
if int(args.task_id) <= 45:
    write_to_eval_paths(args.task_id)

logger.info("Done")
