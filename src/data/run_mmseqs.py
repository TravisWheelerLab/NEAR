""" Running phmmer"""

import argparse
import itertools
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

def create_target_indices():
    for target in range(45):
        targetfasta = f"/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset/targets/targets_{target}.fa"
        cmd = f"mmseqs createdb {targetfasta} /xdisk/twheeler/daphnedemekas/mmseqs_DBs/targetDB{target}"
        out = subprocess.run(cmd, shell=True, check=True,)
        print(out)

        cmd = f"mmseqs createdb {targetfasta} /xdisk/twheeler/daphnedemekas/mmseqs_DBs/targetDB{target}"
        out = subprocess.run(cmd, shell=True, check=True,)
        print(out)

def main(query_fasta, target_index, output_dir):
    cmd = f'mmseqs easy search {query_fasta} {target_index} {output_dir}/alnRes.m8 tmp'
    _ = subprocess.run(cmd, shell=True, check=True,)


def make(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("task_id")

    # args = parser.parse_args()
    create_target_indices()
    for query in range(5):
        print(f"Searching query: {query}")
        make(f"/xdisk/twheeler/daphnedemekas/mmseqs-output/{query}")
        queryfasta = f"/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset/queries/queries_{query}.fa"

        for target in range(45):
            make(f"/xdisk/twheeler/daphnedemekas/mmseqs-output/{target}")

            output_dir = f"/xdisk/twheeler/daphnedemekas/mmseqs-output/{query}/{target}"

            targetindex = f"targetDB{target}"

            main(queryfasta, targetindex, output_dir)
