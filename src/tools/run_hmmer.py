""" Running phmmer"""

import argparse
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def main(query_fasta, target_fasta, outputfile, stdout_file, hmmer_max=False):
    if hmmer_max:
        cmd = f'phmmer --cpu 16 --max --tblout {outputfile} \
            "{query_fasta}" "{target_fasta}"'
    else:
        cmd = f'phmmer --cpu 16 --tblout {outputfile} \
            "{query_fasta}" "{target_fasta}"'
    hmmer = subprocess.run(cmd, shell=True, capture_output=True, check=True,)

    with open(stdout_file, "w") as file:
        file.write(hmmer.stdout.decode("utf-8"))
        file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_fasta")
    parser.add_argument("--target_fasta")
    parser.add_argument("--tbloutfile")
    parser.add_argument("--stdoutfile")
    parser.add_argument("--max", action="store_true")

    args = parser.parse_args()

    main(args.query_fasta, args.target_fasta, args.tbloutfile, args.stdoutfile, args.max)
