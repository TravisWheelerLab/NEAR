import os
from Bio import SearchIO
import pdb
import logging
import itertools
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

train_target_file = open('/xdisk/twheeler/daphnedemekas/target_data/trainfastanames.txt','r')
train_targets = train_target_file.read().splitlines()
val_target_file = open('/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt','r')
val_targets = val_target_file.read().splitlines()

TRAIN_IDX_START =260725
VAL_IDX_START = 141519


def main(task_id):
    """Generate alignments for a given task id"""
    task_id = int(task_id)
    logger.info(f"Task ID: {task_id}")

    ts = list(range(45))
    qs = list(range(5))
    target_queries = list(itertools.product(ts, qs))

    target_filenum = target_queries[task_id - 1][0]
    query_filenum = target_queries[task_id - 1][1]

    print(f"Parsing stdout for q {query_filenum}, t {target_filenum}")
    logger.info(f"Parsing stdout for q {query_filenum}, t {target_filenum}")

    parse_stdout(query_filenum, target_filenum)

def parse_stdout(query_filenum, target_filenum):
    TRAIN_IDX = 0
    VAL_IDX = 0 
    stdout_path = (
        f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/stdouts/{query_filenum}-{target_filenum}.txt"
    )
    dirpath1 = f'/xdisk/twheeler/daphnedemekas/train-alignments/{query_filenum}'
    dirpath2 = f'/xdisk/twheeler/daphnedemekas/train-alignments/{query_filenum}/{target_filenum}'

    if not os.path.exists(dirpath1):
        os.mkdir(dirpath1)
    if not os.path.exists(dirpath2):
        os.mkdir(dirpath2)
    dirpath1 = f'/xdisk/twheeler/daphnedemekas/eval-alignments/{query_filenum}'
    dirpath2 = f'/xdisk/twheeler/daphnedemekas/eval-alignments/{query_filenum}/{target_filenum}'

    if not os.path.exists(dirpath1):
        os.mkdir(dirpath1)
    if not os.path.exists(dirpath2):
        os.mkdir(dirpath2)

    result = SearchIO.parse(stdout_path, "hmmer3-text")

    # result_dict = {}
    # os.mkdir(f'/xdisk/twheeler/daphnedemekas/alignments/{query_filenum}-{target_filenum}')
    for qresult in result:
        # print("Search %s has %i hits" % (qresult.id, len(qresult)))
        query_id = qresult.id
        # result_dict[query_id] = {}
        for idx, hit in enumerate(qresult):
            target_id = hit.id

            if hit.evalue > 10:
                #print("E value > 10, skipping.")
                continue

            # result_dict[query_id][target_id] = []
            for al in range(len(qresult[idx])):
                if target_id in train_targets and hit.evalue < 1:
                    if query_filenum == 0 and target_filenum == 0 and TRAIN_IDX < TRAIN_IDX_START:
                        TRAIN_IDX += 1
                        continue
                    else:
                        alignment_file = open(f'/xdisk/twheeler/daphnedemekas/train-alignments/{query_filenum}/{target_filenum}/{TRAIN_IDX}.txt','w')
                        TRAIN_IDX += 1
                elif target_id in val_targets:
                    if query_filenum == 0 and target_filenum == 0 and VAL_IDX < VAL_IDX_START:
                        VAL_IDX += 1
                        continue
                    else:
                        alignment_file = open(f'/xdisk/twheeler/daphnedemekas/eval-alignments/{query_filenum}/{target_filenum}/{VAL_IDX}.txt','w')
                        VAL_IDX += 1
                else:
                    #print(f"{target_id} not in data")
                    continue

                hsp = qresult[idx][al]
                alignments = hsp.aln
                seq1 = str(alignments[0].seq)
                seq2 = str(alignments[1].seq)
                alignment_file.write(">" + query_id + " & " + target_id + "\n")
                alignment_file.write(seq1 + "\n")
                alignment_file.write(seq2)
                alignment_file.close()

            # result_dict[query_id][target_id].append([seq1, seq2])

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

main(args.task_id)
