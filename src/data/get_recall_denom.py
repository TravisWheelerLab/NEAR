import os
import pickle
import tqdm

all_hits_max_file_4 = "data/evaluationtargetdict"
all_hits_normal_file_4 = "data/evaluationtargetdictnormal"


def load_hmmer_hits(query_id: int = 4):
    """Loads pre-saved hmmer hits dictionaries for a given
    evaluation query id, currently can only be 4 or 0"""
    if query_id == 4:
        with open(all_hits_max_file_4 + ".pkl", "rb") as file:
            all_hits_max_4 = pickle.load(file)
        with open(all_hits_normal_file_4 + ".pkl", "rb") as file:
            all_hits_normal_4 = pickle.load(file)
        return all_hits_max_4, all_hits_normal_4
    else:
        raise Exception(f"No evaluation data for given query id {query_id}")


all_hits_max, _ = load_hmmer_hits(4)
evalue_thresholds = [1e-10, 1e-4, 1e-1, 10]

evalseqs = "/xdisk/twheeler/daphnedemekas/targetdataseqs/eval.txt"

query_sequences = [
    t[:-4]
    for t in os.listdir("/xdisk/twheeler/daphnedemekas/prefilter-output/CPU-5K-20")
]


# with open("target_names.txt", "r") as f:
#     target_sequences = [t.strip("\n") for t in f.readlines()]
# print(f"Found {len(query_sequences)} queries and {len(target_sequences)} targets")


def get_num_decoys(hmmer_hits):
    with open("reversed-target-names.txt", "r") as f:
        target_sequences = [t.strip("\n") for t in f.readlines()]
    print(f"Found {len(query_sequences)} queries and {len(target_sequences)} targets")

    num_pos_per_evalue = [0, 0, 0, 0]
    num_marg_per_evalue = [0, 0, 0, 0]
    num_decoys = [0, 0, 0, 0]

    numhits = len(query_sequences) * len(target_sequences)
    for query in query_sequences:
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
            # numhits += 1
    print(num_pos_per_evalue)
    print(num_marg_per_evalue)
    for i in range(4):
        num_decoys[i] = numhits - num_pos_per_evalue[i] - num_marg_per_evalue[i]
    return num_pos_per_evalue, num_decoys


def get_numpos_per_evalue(hmmer_hits):
    num_pos_per_evalue = [0, 0, 0, 0]
    num_marg_per_evalue = [0, 0, 0, 0]
    num_decoys = [0, 0, 0, 0]

    numhits = len(query_sequences) * len(target_sequences)
    for query in query_sequences:
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
            # numhits += 1
    print(num_pos_per_evalue)
    print(num_marg_per_evalue)
    for i in range(4):
        num_decoys[i] = numhits - num_pos_per_evalue[i] - num_marg_per_evalue[i]
    return num_pos_per_evalue, num_decoys


# print("Getting recall denominator HMMER Max")
# num_pos_per_evaluemax, decoysmax = get_numpos_per_evalue(all_hits_max)
# print("Getting recall denominator HMMER Normal")
# num_pos_per_evaluenormal, decoysnormal = get_numpos_per_evalue(all_hits_normal)

# with open("hmmerrecall.txt", "w") as f:
#     f.write(f"HMMER MAX pos: {num_pos_per_evaluemax}" + "\n")
#     f.write(f"HMMER MAX decoys: {decoysmax}")

#     f.write("\n")
#     f.write(f"HMMER Normal pos: {num_pos_per_evaluenormal}" + "\n")
#     f.write(f"HMMER Normal decoys: {decoysnormal}")

# f.close()


print("Getting num decoys HMMER Max")
num_pos_per_evaluemax, decoysmax = get_num_decoys(all_hits_max)

with open("decoys.txt", "w") as f:
    f.write(f"HMMER MAX pos: {num_pos_per_evaluemax}" + "\n")
    f.write(f"HMMER MAX decoys: {decoysmax}")

f.close()


def prune(results):
    print("Pruning...")
    for query in os.listdir(results):
        if query[:-4] not in query_sequences:
            print(f"Delete this query: {query}")
            os.remove(f"{results}/{query}")


# # mmseqs = '/xdisk/twheeler/daphnedemekas/prefilter-output/mmseqs'
# # knn = '/xdisk/twheeler/daphnedemekas/prefilter-output/knn-for-homology'
# esm = "/xdisk/twheeler/daphnedemekas/prefilter-output/esm"

# # prune(mmseqs)
# # prune(knn)
# # prune(esm)
