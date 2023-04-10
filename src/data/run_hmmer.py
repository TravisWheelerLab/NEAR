""" Running phmmer"""

import argparse
import itertools
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def main(task_id):
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


def main_evaluation(task_id):
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

    hmmer = subprocess.run(
        f'phmmer --cpu 16 --tblout {root}/phmmer_normal_results/{q_fnum}/{t_fnum}/hits.tblout \
            "{data_path}/queries/queries_{q_fnum}.fa" "{data_path}/targets/targets_{t_fnum}.fa"',
        shell=True,
        capture_output=True,
        check=True,
    )
    print(
        "Saving stdout to '%s/phmmer_normal_results/stdouts/%s-%s.txt'", root, q_fnum, t_fnum,
    )

    with open(
        f"/xdisk/twheeler/daphnedemekas/phmmer_normal_results/stdouts/{q_fnum}-{t_fnum}.txt",
        "w",
        encoding="utf-8",
    ) as stdout_file:
        stdout_file.write(hmmer.stdout.decode("utf-8"))
        stdout_file.close()


def run_phmmer_max(q_fnum: int, t_fnum: int):
    """Run phmmer between all queries and targets
    in the query and target files."""
    root = "/xdisk/twheeler/daphnedemekas"
    data_path = "/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset"
    if not os.path.exists(f"{root}/phmmer_max_results/{q_fnum}"):
        os.mkdir(f"{root}/phmmer_max_results/{q_fnum}")
    if not os.path.exists(f"{root}/phmmer_max_results/{q_fnum}/{t_fnum}"):
        os.mkdir(f"{root}/phmmer_max_results/{q_fnum}/{t_fnum}")

    hmmer = subprocess.run(
        f'phmmer --cpu 16 --max --tblout {root}/phmmer_max_results/{q_fnum}/{t_fnum}/hits.tblout \
            "{data_path}/queries/queries_{q_fnum}.fa" "{data_path}/targets/targets_{t_fnum}.fa"',
        shell=True,
        capture_output=True,
        check=True,
    )
    print(
        "Saving stdout to '%s/phmmer_max_results/stdouts/%s-%s.txt'", root, q_fnum, t_fnum,
    )

    with open(
        f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/stdouts/{q_fnum}-{t_fnum}.txt",
        "w",
        encoding="utf-8",
    ) as stdout_file:
        stdout_file.write(hmmer.stdout.decode("utf-8"))
        stdout_file.close()


parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

main_evaluation(args.task_id)
