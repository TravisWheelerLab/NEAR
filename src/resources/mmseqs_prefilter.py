from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.evaluators.uniref_evaluator import (
    get_hmmer_hits,
    recall_and_filtration,
)


def get_mmseqs_hits(mmseqs_filename):
    queries_to_hits = defaultdict(dict)
    with open(mmseqs_filename, "r") as file:
        for line in file:
            query, target, ungapped_score, diagonal = line.split("\t")
            queries_to_hits[query][target] = float(ungapped_score)

    return queries_to_hits


root = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/"

hmmer_hits = get_hmmer_hits(f"{root}/max_hmmer_hits.txt")
score_threshold = 0.0
figure_path = "mmseqs_prefilter_s15.png"
mmseqs_hits = get_mmseqs_hits(f"{root}/mmseqs/prefilter_results.tsv")

# larger score is better
scores = list(mmseqs_hits.values())
# dictionaries
scores = [s1 for s in scores for s1 in s.values()]
comp_func = np.greater_equal

# how many pairwise comparisons did we filter out?
denom = (2000 * 30000) - 2000

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title(f"mmseqs")
ax.plot([0, 100], [100, 0], "k--", linewidth=2)

for color, evalue_threshold in zip(["r", "c", "g"], [1e-10, 1e-1, 1, 10]):
    filtrations = []
    recalls = []
    for threshold in np.linspace(min(scores), 1000, num=10):
        recall, total_hits = recall_and_filtration(
            mmseqs_hits, hmmer_hits, threshold, comp_func, evalue_threshold=evalue_threshold,
        )
        filtration = 100 * (1.0 - (total_hits / denom))
        filtrations.append(filtration)
        recalls.append(recall)
        print(f"{recall:.3f}, {filtration:.3f}, {threshold:.3f}")

    ax.scatter(filtrations, recalls, c=color, marker="o")
    ax.plot(
        filtrations, recalls, f"{color}--", linewidth=2, label=f"eval:{evalue_threshold}",
    )
    ax.set_ylim([-1, 101])
    ax.set_xlim([-1, 101])
    ax.set_xlabel("filtration")
    ax.set_ylabel("recall")

ax.legend()
plt.savefig(f"{figure_path}", bbox_inches="tight")
plt.close()
