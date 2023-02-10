""" Stockholm sequence alignments"""

import subprocess
import os
import logging
from src.data.utils import get_data_from_subset
from tqdm import tqdm

logger = logging.getLogger(__name__)
import itertools


def main(task_id):
    """Generate alignments for a given task id"""
    task_id = int(task_id)
    logger.info(f"Task ID: {task_id}")

    ts = list(range(45))
    qs = list(range(5))
    target_queries = list(itertools.product(ts, qs))

    target_filenum = target_queries[task_id - 1][0]
    query_filenum = target_queries[task_id - 1][1]

    print(f"Running phmmer for q {query_filenum}, t {target_filenum}")
    logger.info(f"Running phmmer for q {query_filenum}, t {target_filenum}")

    run_phmmer(query_filenum, target_filenum)


def run_phmmer(query_filenum: int, target_filenum: int):
    """Run phmmer between all queries and targets
    in the query and target files."""
    if not os.path.exists(f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/{query_filenum}"):
        os.mkdir(f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/{query_filenum}")
    if not os.path.exists(
        f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/{query_filenum}/{target_filenum}"
    ):
        os.mkdir(
            f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/{query_filenum}/{target_filenum}"
        )

    hmmer = subprocess.run(
        f'phmmer --cpu 16 --tblout /xdisk/twheeler/daphnedemekas/phmmer_results/{query_filenum}/{target_filenum}/hits.tblout "/xdisk/twheeler/colligan/uniref/split_subset/queries/queries_{query_filenum}.fa" "/xdisk/twheeler/colligan/uniref/split_subset/targets/targets_{target_filenum}.fa"',
        shell=True,
        capture_output=True,
    )


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

main(args.task_id)
