import pickle
from src.data.benchmarking import get_sorted_pairs
import os
import numpy as np


def load_hmmer_hits(hmmer_hits_file):
    """Loads pre-saved hmmer hits dictionaries for a given
    evaluation query id, currently can only be 4 or 0"""
    with open(hmmer_hits_file + ".pkl", "rb") as file:
        hmmer_hits = pickle.load(file)
    return hmmer_hits


import argparse


def main(hmmer_hits_file, data_savedir):
    hmmer_max, hmmer_normal = load_hmmer_hits(hmmer_hits_file)
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
        np.save(
            f"{data_savedir}/sorted_pairs", np.array(sorted_pairs), allow_pickle=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--all_hits", type=str, help="Path to hmmer hits")
    parser.add_argument("--save_dir", type=str, help="Directory to save results in")
    args = parser.parse_args()
    main(args.all_hits, args.save_dir)
