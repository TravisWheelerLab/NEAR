""" Metrics: module for evaluation metrics and plotting functions """

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json

COLORS = ["r", "c", "g", "k"]


def plot_roc_curve(
    our_hits,
    max_hmmer_hits,
    normalize_embeddings,
    distance_threshold,
    denom,
    figure_path,
    comp_func,
    evalue_thresholds=[1e-10, 1e-1, 1, 10],
):
    """Roc Curve for comparing model hits to the HMMER hits without the prefilter"""
    if normalize_embeddings:
        distances = np.linspace(distance_threshold, 0.999, num=10)
    else:
        distances = np.linspace(0.001, distance_threshold, num=10)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"{os.path.splitext(os.path.basename(figure_path))[0]}")

    # for color, evalue_threshold in zip(["r", "c", "g", "k"], [1e-10, 1e-1, 1, 10]):
    for i, evalue_threshold in enumerate(evalue_thresholds):

        filtrations = []
        recalls = []
        for threshold in tqdm.tqdm(distances):
            recall, total_hits = recall_and_filtration(
                our_hits, max_hmmer_hits, threshold, comp_func, evalue_threshold,
            )

            filtration = 100 * (1.0 - (total_hits / denom))
            filtrations.append(filtration)
            recalls.append(recall)
            print(f"recall: {recall:.3f}, filtration: {filtration:.3f}, threshold: {threshold:.3f}")

        ax.scatter(filtrations, recalls, c=COLORS[i], marker="o")
        ax.plot(filtrations, recalls, f"{COLORS[i]}--", linewidth=2)

    ax.plot([0, 100], [100, 0], "k--", linewidth=2)
    ax.set_ylim([-1, 101])
    ax.set_xlim([-1, 101])
    ax.set_xlabel("filtration")
    ax.set_ylabel("recall")
    plt.savefig(f"{figure_path}", bbox_inches="tight")
    plt.close()


def recall_and_filtration(our_hits, hmmer_hits, distance_threshold, comp_func, evalue_threshold):
    """Function to calculate recall and filtration for a given
    disance threshold"""
    match_count = 0
    our_total_hits = 0
    hmmer_hits_for_our_queries = 0
    # since we sometimes don't have
    # all queries, iterate over the DB in this fashion.
    for query in our_hits:
        if query not in hmmer_hits:
            # we've set an e-value threshold, meaning
            # that this query was never picked up
            print(f"Query {query} not in hmmer_hits.")

        matches = hmmer_hits[query]
        filtered = {}
        for match, evalue in matches.items():
            evalue = float(evalue[0])
            if evalue <= evalue_threshold:
                filtered[match] = evalue

        true_matches = filtered
        hmmer_hits_for_our_queries += len(true_matches)
        our_matches = our_hits[query]
        for match in our_matches:
            if comp_func(our_matches[match], distance_threshold):
                if match in true_matches:
                    # count the matches for each query.
                    # pdb.set_trace()
                    match_count += 1
                our_total_hits += 1

    # total hmmer hits
    denom = hmmer_hits_for_our_queries
    return 100 * (match_count / denom), our_total_hits


def write_output(model_output: dict, hmmer_hits: dict, figure_path: str):
    json = json.dumps(model_output)

    # TODO: make this a function and save it into the ResNet1d thing
    print("Writing output to file...")
    output_path = os.path.join(os.path.dirname(figure_path), "output")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    f = open(f"{output_path}/modelhitsoutput.json", "w")
    f.write(json)
    f.close()
    json = json.dumps(hmmer_hits)
    f = open(f"{output_path}/maxhmmerhits.json", "w")
    f.write(json)
    f.close()
