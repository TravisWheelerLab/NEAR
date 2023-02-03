from src.data.utils import get_data_from_subset
import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def main(task_id):
    task_id = int(task_id)
    logger.info(f"Task ID: {task_id}")

    task_id = task_id - 1

    target_filenum = task_id % 45

    query_filenum = (task_id - 40) % 5

    align(query_filenum, target_filenum)


def align(query_filenum, target_filenum):
    querysequences, targetsequences, _ = get_data_from_subset(
        "uniref/phmmer_results", query_id=query_filenum, file_num=target_filenum
    )
    if not os.path.exists(f"alignments/{query_filenum}"):
        os.mkdir(f"alignments/{query_filenum}")
    if not os.path.exists(f"alignments/{query_filenum}/{target_filenum}"):
        os.mkdir(f"alignments/{query_filenum}/{target_filenum}")
    for queryname, queryseq in querysequences.items():
        for targetname, targetseq in targetsequences.items():
            with open("QT.fa", "w") as outfile:
                outfile.write(">" + queryname + "\n")
                outfile.write(queryseq + "\n")
                outfile.write(">" + targetname + "\n")
                outfile.write(targetseq + "\n")
            alignment = subprocess.run(
                f"mafft QT.fa > alignments/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas",
                shell=True,
                capture_output=True,
            )
            reformat = subprocess.run(
                f"~/easel/miniapps/esl-reformat -o alignments/{query_filenum}/{target_filenum}/{queryname}-{targetname}.stk stockholm alignments/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas",
                shell=True,
                capture_output=True,
            )
            logger.info(reformat)
            os.remove(f"alignments/{query_filenum}/{target_filenum}/{queryname}-{targetname}.fas")


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

main(args.task_id)
