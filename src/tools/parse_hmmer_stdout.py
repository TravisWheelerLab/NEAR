"""Tools to parse stdout of hmmer to generate alignments data"""
import argparse
import logging
import tqdm
from Bio import SearchIO
import os
from src.data.hmmerhits import FastaFile
import itertools

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def parse_stdout(
    stdout_path: str,
    trainpath: str,
    valpath,
    querysequences: dict,
    targetsequences: dict,
    trainseqs, 
    valseqs,
    write_full_seq=True,
):
    """Parse the saved off stdout files from running hmmer

    alignment_file_dir: the directory of the txt file where to save the alignment.
        the alignments will be saved as query_id-target_id

    stdout_path: the path of the hmmer stdout txt file that was saved off from running phmmer

    write_full_seq: This is True if you want to save the original full length sequence along with the alignment

    querysequences, targetsequences: dictionaries of type {id: sequence}. This is needed if write_full_seq = True

    The output files saved in alignment_file_dir will be of structure:

    >query_id & target_id
    aligned_seq1
    aligned_seq2
    full_seq1
    full_seq2
    """

    if write_full_seq:
        assert (
            querysequences is not None
        ), "No querysequences dictionary is given. This is needed to write full sequences"
        assert (
            targetsequences is not None
        ), "No targetsequences dictionary is given. This is needed to write full sequences"

    result = SearchIO.parse(stdout_path, "hmmer3-text")

    for qresult in tqdm.tqdm(result):
        query_id = qresult.id
        for hit in qresult:
            target_id = hit.id

            if target_id in trainseqs:
                if hit.evalue > 1:
                    continue

                alignment_file_path = f"{trainpath}/{query_id}-{target_id}.txt"

            elif target_id in valseqs:
                if hit.evalue > 10:
                    continue
                alignment_file_path = f"{valpath}/{query_id}-{target_id}.txt"

            with open(alignment_file_path, "w") as alignment_file:
                for hsp in hit:
                    alignments = hsp.aln
                    seq1 = str(alignments[0].seq)
                    seq2 = str(alignments[1].seq)
                    alignment_file.write(">" + query_id + " & " + target_id + "\n")
                    alignment_file.write(seq1 + "\n")
                    alignment_file.write(seq2 + "\n")
                    if write_full_seq:
                        fullseq1 = querysequences[query_id]
                        fullseq2 = targetsequences[target_id]
                        alignment_file.write(fullseq1 + "\n")
                        alignment_file.write(fullseq2 + "\n")


def parse(task_id):
    task_id = int(task_id)
    logger.info("Task ID: %s", task_id)

    targets = list(range(45))
    queries = list(range(4))
    target_queries = list(itertools.product(targets, queries))

    target_filenum = target_queries[task_id - 1][0]
    query_filenum = target_queries[task_id - 1][1]

    train_root = '/xdisk/twheeler/daphnedemekas/train-alignments'
    val_root = '/xdisk/twheeler/daphnedemekas/eval-alignments'
    stdout_path = f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/{query_filenum}-{target_filenum}.txt"
    dirpath1 = f"{train_root}/{query_filenum}"
    trainpath = f"{train_root}/{query_filenum}/{target_filenum}"

    queryfile = f"uniref/split_subset/queries/queries_{query_filenum}.fa"
    queryfasta = FastaFile(queryfile)
    querysequences = queryfasta.data
    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{target_filenum}.fa")
    targetsequences = targetfasta.data

    if not os.path.exists(dirpath1):
        os.mkdir(dirpath1)
    if not os.path.exists(trainpath):
        os.mkdir(trainpath)


    dirpath1 = f"{val_root}/{query_filenum}"
    valpath = f"{val_root}/{query_filenum}/{target_filenum}"

    if not os.path.exists(dirpath1):
        os.mkdir(dirpath1)
    if not os.path.exists(valpath):
        os.mkdir(valpath)

    trainseqs = '/xdisk/twheeler/daphnedemekas/targetdataseqs/train.txt'
    evalseqs = '/xdisk/twheeler/daphnedemekas/targetdataseqs/eval.txt'


    with open(trainseqs, "r") as train_target_file:
        train_targets = [t.strip("\n") for t in train_target_file.readlines()]
        print(f"Found {len(train_targets)} train targets")
    with open(evalseqs, "r") as val_target_file:
        val_targets = [t.strip("\n") for t in val_target_file.readlines()]
        print(f"Found {len(val_targets)} val targets")
    parse_stdout(stdout_path=stdout_path, trainpath, valpath, querysequences, targetsequences, train_targets, val_targets)


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stdout_path")
    parser.add_argument("save_dir")
    parser.add_argument("query_fasta")
    parser.add_argument("target_fasta")
    args = parser.parse_args()

    queryfasta = FastaFile(args.query_fasta)
    querysequences = queryfasta.data
    targetfasta = FastaFile(args.target_fasta)
    targetsequences = targetfasta.data

    args = parser.parse_args()











    parse_stdout(args.stdout_path, args.save_dir, querysequences, targetsequences)
