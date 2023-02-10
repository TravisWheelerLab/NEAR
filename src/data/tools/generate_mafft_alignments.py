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

    print(f"Calling alignment for q {query_filenum}, t {target_filenum}")

    align(query_filenum, target_filenum)


def align(query_filenum: int, target_filenum: int):
    """Generates MAFFT alignments between all queries and targets
    in the query and target files and then converts the alignment
    files into stockholm alignments."""
    querysequences, targetsequences, _ = get_data_from_subset(
        "uniref/phmmer_results", query_id=query_filenum, file_num=target_filenum
    )
    if not os.path.exists(f"/xdisk/twheeler/daphnedemekas/stk_alignments/{query_filenum}"):
        os.mkdir(f"/xdisk/twheeler/daphnedemekas/stk_alignments/{query_filenum}")
    if not os.path.exists(
        f"/xdisk/twheeler/daphnedemekas/stk_alignments/{query_filenum}/{target_filenum}"
    ):

        os.mkdir(f"/xdisk/twheeler/daphnedemekas/stk_alignments/{query_filenum}/{target_filenum}")
    else:
        print("END")
        return
    if not os.path.exists(f"/tmp/{query_filenum}"):
        os.mkdir(f"/tmp/{query_filenum}")
    if not os.path.exists(f"/tmp/{query_filenum}/{target_filenum}"):
        os.mkdir(f"/tmp/{query_filenum}/{target_filenum}")
    for queryname, queryseq in tqdm(querysequences.items()):
        t = 0
        for targetname, targetseq in tqdm(targetsequences.items()):
            t += 1

            if os.path.exists(
                f"/xdisk/twheeler/daphnedemekas/stk_alignments/{query_filenum}/{target_filenum}/{queryname}-{targetname}.stk"
            ):
                continue
            else:
                # print(f'generating {t} / {len(targetsequences)} ')
                with open(f"/tmp/{query_filenum}/{target_filenum}/QT.fa", "w") as outfile:
                    outfile.write(">" + queryname + "\n")
                    outfile.write(queryseq + "\n")
                    outfile.write(">" + targetname + "\n")
                    outfile.write(targetseq + "\n")
                mafft = subprocess.run(
                    f"mafft /tmp/{query_filenum}/{target_filenum}/QT.fa > /tmp/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas",
                    shell=True,
                    capture_output=True,
                )
                if mafft.returncode != 0:
                    mafft = subprocess.run(
                        f"mafft --anysymbol /tmp/{query_filenum}/{target_filenum}/QT.fa > /tmp/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas",
                        shell=True,
                        capture_output=True,
                    )
                    if mafft.returncode != 0:
                        print(mafft)
                        raise
                reformat = subprocess.run(
                    f"~/easel/miniapps/esl-reformat -o /xdisk/twheeler/daphnedemekas/stk_alignments/{query_filenum}/{target_filenum}/{queryname}-{targetname}.stk stockholm /tmp/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas",
                    shell=True,
                    capture_output=True,
                )
                if reformat.returncode != 0:
                    print(reformat)
                    raise

            os.remove(f"/tmp/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas")
            os.remove(f"/tmp/{query_filenum}/{target_filenum}/QT.fa")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

main(args.task_id)
