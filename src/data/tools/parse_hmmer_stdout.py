"""Tools to parse stdout of hmmer to generate alignments data"""
import argparse
import itertools
import logging
import os
import tqdm
import yaml
from Bio import SearchIO
from src.data.hmmerhits import FastaFile

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open(config["traintargetspath"], "r") as train_target_file:
    train_targets = [t.strip("\n") for t in train_target_file.readlines()]
    print(f"Found {len(train_targets)} train targets")
with open(config["evaltargetspath"], "r") as val_target_file:
    val_targets = [t.strip("\n") for t in val_target_file.readlines()]
    print(f"Found {len(val_targets)} val targets")


def parse_by_task(task_id=None):
    """Generate alignments for a given task id"""

    task_id = int(task_id)
    logger.info("Task ID: %s", task_id)

    targets = list(range(45))
    queries = list(range(4))
    target_queries = list(itertools.product(targets, queries))

    target_filenum = target_queries[task_id - 1][0]
    query_filenum = target_queries[task_id - 1][1]

    print("Parsing stdout for q %s, t %s", query_filenum, target_filenum)
    logger.info("Parsing stdout for q %s, t %s", query_filenum, target_filenum)

    parse_stdout(query_filenum, target_filenum)
    

def parse_stdout(q_fnum, t_fnum):
    """Parse the saved off stdout files from running hmmer"""
    train_idx = 0
    val_idx = 0
    train_root = config["trainalignmentspath"]
    val_root = config["evalalignmentspath"]
    stdout_path = f"{config['hmmer_stdout_path']}/{q_fnum}-{t_fnum}.txt"
    dirpath1 = f"{train_root}/{q_fnum}"
    dirpath2 = f"{train_root}/{q_fnum}/{t_fnum}"

    queryfile = f"uniref/split_subset/queries/queries_{q_fnum}.fa"
    queryfasta = FastaFile(queryfile)
    querysequences = queryfasta.data
    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{t_fnum}.fa")
    targetsequences = targetfasta.data

    if not os.path.exists(dirpath1):
        os.mkdir(dirpath1)
    if not os.path.exists(dirpath2):
        os.mkdir(dirpath2)
    dirpath1 = f"{val_root}/{q_fnum}"
    dirpath2 = f"{val_root}/{q_fnum}/{t_fnum}"

    if not os.path.exists(dirpath1):
        os.mkdir(dirpath1)
    if not os.path.exists(dirpath2):
        os.mkdir(dirpath2)

    result = SearchIO.parse(stdout_path, "hmmer3-text")

    for qresult in tqdm.tqdm(result):
        query_id = qresult.id
        for hit in qresult:
            target_id = hit.id

            if hit.evalue > 1:
                continue

            if target_id in train_targets:
                alignment_file_path = f"{train_root}/{q_fnum}/{t_fnum}/{train_idx}.txt"
                train_idx += 1
            elif target_id in val_targets:
                alignment_file_path = f"{val_root}/{q_fnum}/{t_fnum}/{val_idx}.txt"
                val_idx += 1
            else:
                continue
            with open(alignment_file_path, "w") as alignment_file:
                for hsp in hit:
                    alignments = hsp.aln
                    seq1 = str(alignments[0].seq)
                    seq2 = str(alignments[1].seq)
                    alignment_file.write(">" + query_id + " & " + target_id + "\n")
                    alignment_file.write(seq1 + "\n")
                    alignment_file.write(seq2 + "\n")
                    fullseq1 = querysequences[query_id]
                    fullseq2 = targetsequences[target_id]
                    alignment_file.write(fullseq1 + "\n")
                    alignment_file.write(fullseq2 + "\n")



def get_evaluation_alignment_data():
    queryfasta = FastaFile("uniref/split_subset/queries/queries_4.fa")
    querysequences = queryfasta.data


    for t_fnum in tqdm.tqdm(range(45)):
        stdout_path = f"{config['hmmer_stdout_path']}/4-{t_fnum}.txt"
        result = SearchIO.parse(stdout_path, "hmmer3-text")
        targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{t_fnum}.fa")
        targetsequences = targetfasta.data
        for qresult in tqdm.tqdm(result):
            query_id = qresult.id
            for hit in qresult:
                target_id = hit.id

                if hit.evalue > 1:
                    continue
                if target_id in val_targets:
                    for hsp in hit:
                        alignments = hsp.aln
                        seq1 = str(alignments[0].seq)
                        seq2 = str(alignments[1].seq)
                        fullseq1 = querysequences[query_id]
                        fullseq2 = targetsequences[target_id]

                        if len(seq1) / len(fullseq1) >= 0.9 and len(seq2) / len(fullseq2) >= 0.9:
                            alignment_file_dir = f"/xdisk/twheeler/daphnedemekas/query4-alignments/over-90"
                        elif len(seq1) / len(fullseq1) <= 0.5 or len(seq2) / len(fullseq2) <= 0.5:
                            alignment_file_dir = f"/xdisk/twheeler/daphnedemekas/query4-alignments/under-50"
                        else:
                            alignment_file_dir = f"/xdisk/twheeler/daphnedemekas/query4-alignments/between"

                        queries = open(f'{alignment_file_dir}/queries.fa','a')
                        queries.write(">" + query_id + " " + "\n")
                        queries.write(fullseq1 + "\n")
                        queries.close()
                        targets = open(f'{alignment_file_dir}/targets.fa','a')
                        targets.write(">" + target_id + " " + "\n")
                        targets.write(fullseq2 + "\n")
                        targets.close()

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("task_id")
    # args = parser.parse_args()

    # parse_by_task(args.task_id)

    get_evaluation_alignment_data()
