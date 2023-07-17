import pickle
from src.data.benchmarking import get_sorted_pairs
import os
import numpy as np

all_hits_max_file_4 = "data/evaluationtargetdict"
all_hits_normal_file_4 = "data/evaluationtargetdictnormal"

data_savedir = "/xdisk/twheeler/daphnedemekas/hmmer-normal"


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


hmmer_normal, hmmer_max = load_hmmer_hits()

all_pairs = []
all_scores = []
all_e_values = []
all_biases = []
similariites = []
for query, targethits in hmmer_normal.items():
    for target, scores in targethits.items():
        score = scores[1]
        all_pairs.append((query, target))
        all_scores.append(score)
        all_e_values.append(hmmer_max[query][target][0])
        all_biases.append(hmmer_max[query][target][2])

print("Sorting pairs...")
sorted_pairs = get_sorted_pairs(all_scores, all_pairs)

if data_savedir is not None:
    if not os.path.exists(data_savedir):
        os.mkdir(data_savedir)
    print(f"Saving to {data_savedir}:")
    np.save(
        f"{data_savedir}/similarities",
        np.array(all_scores, dtype="half"),
        allow_pickle=True,
    )
    np.save(
        f"{data_savedir}/all_biases",
        np.array(all_biases, dtype="half"),
        allow_pickle=True,
    )
    np.save(
        f"{data_savedir}/all_e_values",
        np.array(all_e_values, dtype="half"),
        allow_pickle=True,
    )
    np.save(f"{data_savedir}/sorted_pairs", np.array(sorted_pairs), allow_pickle=True)
