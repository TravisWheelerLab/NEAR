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

with open(config["traintargetspath"], "rb") as train_target_file:
    train_targets = train_target_file.read().splitlines()
with open(config["evaltargetspath"], "rb") as val_target_file:
    val_targets = val_target_file.read().splitlines()


def main(task_id=None):
    """Generate alignments for a given task id"""
    if task_id is None:
        for q in [1, 2, 3]:
            t = 0
            print("Parsing stdout for q %s, t %s", q, t)
            parse_stdout(q, t)
        return None

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

    # if len(os.listdir(dirpath2)) != 0:
    #     print("Already have these alignments")
    #     sys.exit()
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
        # print("Search %s has %i hits" % (qresult.id, len(qresult)))
        query_id = qresult.id
        # result_dict[query_id] = {}
        for hit in qresult:
            target_id = hit.id

            if hit.evalue > 10:
                # print("E value > 10, skipping.")
                continue

            # result_dict[query_id][target_id] = []
            if target_id in train_targets and hit.evalue < 1:
                # if query_id != 0:
                alignment_file_path = f"{train_root}/{q_fnum}/{t_fnum}/{train_idx}.txt"
                train_idx += 1
            elif target_id in val_targets:
                # if query_id == 0:
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    args = parser.parse_args()

    main(args.task_id)
