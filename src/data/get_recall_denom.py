import os
import pickle
import tqdm
from src.data.hmmerhits import FastaFile


def get_numpos_per_evalue(hmmer_hits, query_sequences, target_sequences):
    num_pos_per_evalue = [0, 0, 0, 0]
    num_marg_per_evalue = [0, 0, 0, 0]
    num_decoys = [0, 0, 0, 0]

    numhits = len(query_sequences) * len(target_sequences)
    for query in tqdm.tqdm(query_sequences):
        if query not in hmmer_hits:
            continue

        query_data = hmmer_hits[query]
        for target in target_sequences:
            if target not in query_data:
                continue
            data = query_data[target]
            evalue = data[0]
            if evalue < evalue_thresholds[3]:  # evalue < 10
                num_pos_per_evalue[3] += 1
                if evalue < evalue_thresholds[2]:  # evalue < e-1
                    num_pos_per_evalue[2] += 1

                    if evalue < evalue_thresholds[1]:
                        num_pos_per_evalue[1] += 1

                        if evalue < evalue_thresholds[0]:
                            num_pos_per_evalue[0] += 1
                        else:
                            num_marg_per_evalue[0] += 1
                    else:
                        num_marg_per_evalue[0] += 1
                        num_marg_per_evalue[1] += 1

                else:  # e-1 < evalue < 10
                    num_marg_per_evalue[0] += 1
                    num_marg_per_evalue[1] += 1
                    num_marg_per_evalue[2] += 1
            else:
                num_marg_per_evalue[0] += 1
                num_marg_per_evalue[1] += 1
                num_marg_per_evalue[2] += 1
                num_marg_per_evalue[3] += 1

    for i in range(4):
        num_decoys[i] = numhits - num_pos_per_evalue[i] - num_marg_per_evalue[i]
    return num_pos_per_evalue, num_decoys


def prune(results):
    print("Pruning...")
    for query in os.listdir(results):
        if query[:-4] not in query_sequences:
            print(f"Delete this query: {query}")
            os.remove(f"{results}/{query}")


all_hits_max_file_4 = "data/hmmerhits-masked-dict"

with open(all_hits_max_file_4 + ".pkl", "rb") as file:
    all_hits_max = pickle.load(file)

targetfile = FastaFile("/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked.fa")
queriesfile = FastaFile(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/queries-masked.fa"
)

query_sequences = queriesfile.data
target_sequences = targetfile.data

evalue_thresholds = [1e-10, 1e-4, 1e-1, 10]

num_pos_per_evalue, numdecoys = get_numpos_per_evalue(
    all_hits_max, query_sequences, target_sequences
)

print(f"Num pos per evalue: {num_pos_per_evalue}")
print(f"Num decoys: {numdecoys}")

with open("decoys.txt", "w") as f:
    f.write(f"HMMER MAX pos: {num_pos_per_evalue}" + "\n")
    f.write(f"HMMER MAX decoys: {numdecoys}")

f.close()
