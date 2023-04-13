""" Running phmmer"""

import argparse
import itertools
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def run_training_task(task_id):
    """Generate alignments for a given task id"""
    task_id = int(task_id)
    logger.info("Task ID: %s", task_id)

    targets = list(range(45))
    queries = list(range(5))
    target_queries = list(itertools.product(targets, queries))

    target_filenum = target_queries[task_id - 1][0]
    query_filenum = target_queries[task_id - 1][1]

    print(f"Running phmmer for q {query_filenum}, t {target_filenum}")
    logger.info("Running phmmer for q %s, t %s", query_filenum, target_filenum)

    run_phmmer(query_filenum, target_filenum)
    run_phmmer_max(query_filenum, target_filenum)


def run_evaluation_task(task_id):
    """Generate alignments for a given task id"""
    target_filenum = int(task_id) - 1
    logger.info("Target File: %s", target_filenum)

    query_filenum = 4

    print(f"Running phmmer for q {query_filenum}, t {target_filenum}")

    run_phmmer(query_filenum, target_filenum)
    run_phmmer_max(query_filenum, target_filenum)

def run_phmmer(q_fnum: int, t_fnum: int):
    """Run phmmer between all queries and targets
    in the query and target files."""
    root = "/xdisk/twheeler/daphnedemekas"
    data_path = "/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset"
    if not os.path.exists(f"{root}/phmmer_normal_results/{q_fnum}"):
        os.mkdir(f"{root}/phmmer_normal_results/{q_fnum}")
    if not os.path.exists(f"{root}/phmmer_normal_results/{q_fnum}/{t_fnum}"):
        os.mkdir(f"{root}/phmmer_normal_results/{q_fnum}/{t_fnum}")

    query_fasta = f"{data_path}/queries/queries_{q_fnum}.fa"
    target_fasta = f"{data_path}/targets/targets_{t_fnum}.fa"
    outputfile = f"{root}/phmmer_normal_results/{q_fnum}/{t_fnum}/hits.tblout"
    stdout_file = f"/xdisk/twheeler/daphnedemekas/phmmer_normal_results/stdouts/{q_fnum}-{t_fnum}.txt"

    main(query_fasta, target_fasta, outputfile, stdout_file)


def run_phmmer_max(q_fnum: int, t_fnum: int):
    """Run phmmer between all queries and targets
    in the query and target files."""
    root = "/xdisk/twheeler/daphnedemekas"
    data_path = "/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset"
    if not os.path.exists(f"{root}/phmmer_max_results/{q_fnum}"):
        os.mkdir(f"{root}/phmmer_max_results/{q_fnum}")
    if not os.path.exists(f"{root}/phmmer_max_results/{q_fnum}/{t_fnum}"):
        os.mkdir(f"{root}/phmmer_max_results/{q_fnum}/{t_fnum}")

    query_fasta = f"{data_path}/queries/queries_{q_fnum}.fa"
    target_fasta = f"{data_path}/targets/targets_{t_fnum}.fa"
    outputfile = f"{root}/phmmer_max_results/{q_fnum}/{t_fnum}/hits.tblout"
    stdout_file = f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/stdouts/{q_fnum}-{t_fnum}.txt"

    main(query_fasta, target_fasta, outputfile, stdout_file, hmmer_max = True)


def main(query_fasta, target_fasta, outputfile, stdout_file, hmmer_max = False):
    if hmmer_max:
        cmd = f'phmmer --cpu 16 --max --tblout {outputfile} \
            "{query_fasta}" "{target_fasta}"'
    else:
        cmd = f'phmmer --cpu 16 --tblout {outputfile} \
            "{query_fasta}" "{target_fasta}"'
    hmmer = subprocess.run(cmd,
        shell=True,
        capture_output=True,
        check=True,
    )

    with open(stdout_file,"w") as file:
        file.write(hmmer.stdout.decode("utf-8"))
        file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_fasta")
    parser.add_argument("--target_fasta")
    parser.add_argument("--tbloutfile")
    parser.add_argument("--stdoutfile")
    parser.add_argument("--max", action = "store_true")

    args = parser.parse_args()

    main(args.query_fasta, args.target_fasta, args.tbloutfile, args.stdoutfile, args.max)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("task_id")
    # args = parser.parse_args()

    # run_training_task(args.task_id)
    # run_evaluation_task(args.task_id)


