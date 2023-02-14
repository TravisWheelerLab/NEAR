import os
from Bio import SearchIO
import pdb


TRAIN_QUERY_FILENUMS = [0,1,2,3]
VAL_QUERY_FILENUMS = [4]
target_filenums = list(range(45))

train_target_file = open('/xdisk/twheeler/daphnedemekas/target_data/trainfastanames.txt','r')
train_targets = train_target_file.read().splitlines()
val_target_file = open('/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt','r')
val_targets = val_target_file.read().splitlines()

query_filenum = TRAIN_QUERY_FILENUMS[0]
target_filenum = target_filenums[0]
TRAIN_IDX =0
VAL_IDX = 0

for query_filenum in [0,1,2,3,4]:
    print(f"Query filenum: {query_filenum}")
    for target_filenum in range(45):
        print(f"Target filenum: {target_filenum}")

        stdout_path = (
            f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/stdouts/{query_filenum}-{target_filenum}.txt"
        )

        result = SearchIO.parse(stdout_path, "hmmer3-text")

        # result_dict = {}
        # os.mkdir(f'/xdisk/twheeler/daphnedemekas/alignments/{query_filenum}-{target_filenum}')
        for qresult in result:
            # print("Search %s has %i hits" % (qresult.id, len(qresult)))
            query_id = qresult.id
            print(f"QueryID: {query_id}")
            # result_dict[query_id] = {}
            for idx, hit in enumerate(qresult):
                target_id = hit.id
                # result_dict[query_id][target_id] = []
                for al in range(len(qresult[idx])):
                    if target_id in train_targets:
                        alignment_file = open(f'/xdisk/twheeler/daphnedemekas/train-alignments/{TRAIN_IDX}.txt','w')
                        TRAIN_IDX += 1
                    elif target_id in val_targets:
                        alignment_file = open(f'/xdisk/twheeler/daphnedemekas/eval-alignments/{VAL_IDX}.txt','w')
                        VAL_IDX += 1
                    else:
                        print(f"{target_id} not in data")
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

