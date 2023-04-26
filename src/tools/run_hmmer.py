""" Running phmmer"""

import argparse
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


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
    stdout_file = (
        f"/xdisk/twheeler/daphnedemekas/phmmer_normal_results/stdouts/{q_fnum}-{t_fnum}.txt"
    )

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

    main(query_fasta, target_fasta, outputfile, stdout_file, hmmer_max=True)


def main(query_fasta, target_fasta, outputfile, stdout_file, hmmer_max=False):
    if hmmer_max:
        cmd = f'phmmer --cpu 16 --max --tblout {outputfile} \
            "{query_fasta}" "{target_fasta}"'
    else:
        cmd = f'phmmer --cpu 16 --tblout {outputfile} \
            "{query_fasta}" "{target_fasta}"'
    hmmer = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        check=True,
    )

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
