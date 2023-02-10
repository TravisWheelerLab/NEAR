import os
from Bio import SearchIO
import pdb

query_filenum = 0
target_filenum = 0

stdout_path = (
    f"/xdisk/twheeler/daphnedemekas/phmmer_max_results/stdouts/{query_filenum}-{target_filenum}.txt"
)

result = SearchIO.parse(stdout_path, "hmmer3-text")


START_IDX = 0
# result_dict = {}
# os.mkdir(f'/xdisk/twheeler/daphnedemekas/alignments/{query_filenum}-{target_filenum}')
for qresult in result:
    # print("Search %s has %i hits" % (qresult.id, len(qresult)))
    query_id = qresult.id
    # result_dict[query_id] = {}
    for idx, hit in enumerate(qresult):
        target_id = hit.id
        # result_dict[query_id][target_id] = []
        for al in range(len(qresult[idx])):
            START_IDX += 1
            # alignment_file = open(f'/xdisk/twheeler/daphnedemekas/alignments/{query_filenum}-{target_filenum}/{START_IDX}.txt','w')

            hsp = qresult[idx][al]
            alignments = hsp.aln
            seq1 = str(alignments[0].seq)
            seq2 = str(alignments[1].seq)
            # alignment_file.write(seq1 + "\n")
            # alignment_file.write(seq2)
            # alignment_file.close()

            # result_dict[query_id][target_id].append([seq1, seq2])
print(START_IDX)
pdb.set_trace()
