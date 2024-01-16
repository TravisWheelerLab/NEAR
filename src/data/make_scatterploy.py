import os
import pickle
import tqdm

all_hits_max_file_4 = "data/hmmerhits-masked-dict"


def load_hmmer_hits(query_id: int = 4):
    """Loads pre-saved hmmer hits dictionaries for a given
    evaluation query id, currently can only be 4 or 0"""
    if query_id == 4:
        with open(all_hits_max_file_4 + ".pkl", "rb") as file:
            all_hits_max_4 = pickle.load(file)
        return all_hits_max_4


query_lengths_file = "data/query-lengths-masked.pkl"

hmmer_data = load_hmmer_hits(4)
near_data_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/CPU-5K-20-masked"


evalues = []
similarities = []
with open(query_lengths_file, "rb") as f:
    query_lengths = pickle.load(f)

for file in tqdm.tqdm(os.listdir(near_data_dir)):
    queryname = file.strip(".txt")
    querylength = query_lengths[queryname]
    if queryname not in hmmer_data:
        continue
    with open(os.path.join(near_data_dir, file), "r") as f:
        for line in file:
            if "Distance" in line:
                continue
            target = line.split()[0].strip("\n").strip(".pt")
            if target not in hmmer_data[queryname]:
                continue
            similarity = float(line.split()[1].strip("\n")) * 100 / querylength
            evalue = hmmer_data[queryname][target][0]
            evalues.append(evalue)
            similarities.append(similarity)

with open("hmmerevalues.txt", "w") as f:
    for evalue in evalues:
        f.write(f"{evalue}\n")

with open("near-similarities.txt", "w") as f:
    for similarity in similarities:
        f.write(f"{similarity}\n")

import matplotlib.pyplot as plt

plt.scatter(evalues, similarities)
plt.xlabel("HMMER E-value")
plt.ylabel("NEAR Similarity")
plt.savefig("ResNet1d/results/evalue-similarity-scatter.png")
plt.close()
