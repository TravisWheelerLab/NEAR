"""File with functions to evaluate model """
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data.utils import get_evaluation_data

COLORS = ["r", "c", "g", "k"]


def plot_mean_e_values(
    distance_list: list,
    e_values_list: list,
    biases: list,
    min_threshold: int = 0,
    max_threshold: int = 300,
    outputfilename: str = "evaluemeans",
    plot_stds: bool = True,
    plot_lengths: bool = True,
    title: str = "",
    scatter_size: int = 1,
):
    """This plots the correlation between the average
    hmmer e values to the model's similarity score
    at different thresholds of similarity

    A good result will show that low evalues
    are correlated with high similarity
    so that the average e value decreases as
    the similarity threshold increases"""
    plt.clf()
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    all_distance = np.array(distance_list)
    all_e_values = np.array(e_values_list)

    means = []
    stds = []
    lengths = []
    mean_bias = []
    for threshold in thresholds:
        idx = np.where(all_distance > threshold)[0]
        mean = np.mean(np.ma.masked_invalid(np.log10(all_e_values[idx])))
        std = np.std(np.ma.masked_invalid(np.log10(all_e_values[idx])))
        bias = np.mean(biases[idx])
        means.append(mean)
        stds.append(std)
        length = len(idx)
        lengths.append(length)
        mean_bias.append(bias)
    lengths = np.log(lengths)
    plt.scatter(thresholds, means, c=mean_bias, cmap="Greens", s=scatter_size)
    plt.plot(thresholds, means)
    if plot_stds:
        plt.fill_between(
            thresholds,
            np.array(means) - np.array(stds) / 2,
            np.array(means) + np.array(stds) / 2,
            alpha=0.5,
        )
    if plot_lengths:
        plt.fill_between(
            thresholds,
            np.array(means) - np.array(lengths),
            np.array(means) + np.array(lengths),
            alpha=0.5,
            color="orange",
        )
    plt.title(title)
    plt.ylabel("Log E value means")
    plt.xlabel("Similarity Threshold")
    plt.savefig(f"ResNet1d/eval/{outputfilename}.png")


def plot_roc_curve(
    figure_path: str,
    numpos_per_evalue: list,
    evalue_thresholds: list = [1e-10, 1e-1, 1, 10],
    filename: str = "data.txt",
):
    """This plots the ROC curve comparing the
    recall and filtration of this model to hmmer hits
    The filtration is currently being thresholded to be
    greater than 80%
    Before running this function you need to construct
    a path (argument filename) that counts the recall
    and filtration line by line for a given evalue threshold

    The numpos arguments refer to the number of positives
    for a given evalue threshold"""
    _, axis = plt.subplots(figsize=(10, 10))
    axis.set_title(f"{os.path.splitext(os.path.basename(figure_path))[0]}")
    with open(filename, "r", encoding="utf-8") as datafile:
        lines = datafile.readlines()
        numhits = len(lines)

        for i, evalue_threshold in enumerate(evalue_thresholds):
            print(f"Evalue threshold: {evalue_threshold}")

            filtrations = []
            recalls = []

            num_positives = 0
            num_decoys = 0
            recall = None
            filtration = None

            for idx, line in enumerate(lines):
                line = line.split()
                classname = line[3 + i]
                if classname == "P":
                    num_positives += 1
                    recall = num_positives / numpos_per_evalue[i]
                elif classname == "D":
                    num_decoys += 1
                    filtration = num_decoys / numhits

                if recall is not None and filtration is not None and (1 - filtration) > 0.8:
                    if idx % 1000 == 0:
                        print(
                            f"num_Ps: {num_positives},  num_Ds: {num_decoys},  \
                                recall: {100*recall:.3f}, filtration: {100*(1-filtration):.3f}"
                        )
                        filtrations.append(100 * (1 - filtration))
                        recalls.append(100 * recall)

    axis.plot(filtrations, recalls, f"{COLORS[i]}--", linewidth=2, label=evalue_threshold)

    axis.plot([80, 100], [100, 0], "k--", linewidth=2)

    axis.set_xlabel("filtration")
    axis.set_ylabel("recall")
    plt.legend()
    plt.savefig(f"{figure_path}", bbox_inches="tight")
    plt.close()


def get_scores_and_pairs(modelhitsfile: str, hmmerhits: dict) -> Tuple[list, list]:
    """parses the output file from our model
    and returns a list of scores and query-target
    pairs for the results that are also in hmmer hits"""
    all_scores = []
    all_pairs = []
    for queryhits in os.listdir(modelhitsfile):
        queryname = queryhits.strip(".txt")
        if queryname not in hmmerhits.keys():
            print(f"Query {queryname} not in hmmer hits")
            continue

        with open(f"{modelhitsfile}/{queryhits}", "r", encoding="utf-8") as file:
            scores = file.readlines()

            for line in scores:
                if "Distance" in line:
                    continue
                target = line.split()[0].strip("\n")

                score = float(line.split()[1].strip("\n"))
                all_scores.append(score)

                all_pairs.append((queryname, target))
    return all_scores, all_pairs


def write_datafile(
    pairs: list,
    hmmerhits: dict,
    evalue_thresholds: list = [1e-10, 1e-1, 1, 10],
    filename: str = "data.txt",
) -> list:
    """This opens a data file and counts the recall as it descends through the results
    Returns the number of positives for 4 evalue thresholds
    TODO: This code can be improved /generalized"""
    numpos_per_evalue = [0, 0, 0, 0]
    with open(filename, "w", encoding="utf-8") as datafile:
        for pair in pairs:
            query, target = pair[0], pair[1]

            evalue = None
            if target not in hmmerhits[query].keys():
                classname1 = classname2 = classname3 = classname4 = "D"
            else:
                evalue = hmmerhits[query][target][0]
                classname1 = classname2 = classname3 = classname4 = "M"

                if evalue < evalue_thresholds[3]:
                    classname4 = "P"
                    numpos_per_evalue[3] += 1
                    if evalue < evalue_thresholds[2]:
                        classname3 = "P"
                        numpos_per_evalue[2] += 1
                        if evalue < evalue_thresholds[1]:
                            classname2 = "P"
                            numpos_per_evalue[1] += 1
                            if evalue < evalue_thresholds[0]:
                                classname1 = "P"
                                numpos_per_evalue[0] += 1

            datafile.write(
                f"{query}          {target}          {evalue}          \
                    {classname1}          {classname2}          \
                        {classname3}          {classname4}"
                + "\n"
            )

    return numpos_per_evalue


def generate_roc(filename: str, modelhitsfile: str, hmmerhits: dict, figure_path: str):
    """Pipeline to write data to file and generate the ROC plot
    This will then delete the file as well as its massive and not useful"""
    scores, pairs = get_scores_and_pairs(modelhitsfile, hmmerhits)
    print("Got data")
    sortedidx = np.argsort(scores)[::-1]
    pairs = np.array(pairs)[sortedidx]
    numpos_per_evalue = write_datafile(
        pairs, hmmerhits, evalue_thresholds=[1e-10, 1e-1, 1, 10], filename=filename
    )
    print("Wrote files")
    plot_roc_curve(figure_path, numpos_per_evalue, filename=filename)
    os.remove(filename)


def get_outliers_and_inliers(all_similarities: list, all_e_values: list, all_targets: list):
    """Writes outliers and inliers to a text file
    where outliers are defined as our hits that have large similairty
    but high e value"""
    d_idxs = np.where(all_similarities > 1000)[0]

    e_vals = all_e_values[d_idxs]

    outliers = np.where(e_vals > 1)[0]

    outlier_idx = d_idxs[outliers]

    with open("outliers.txt", "w", encoding="utf-8") as outliers_file:
        for idx in outlier_idx:
            pair = all_targets[idx]
            outliers_file.write("Query" + "\n" + str(pair[0]) + "\n")
            outliers_file.write(querysequences_max[pair[0]] + "\n")
            outliers_file.write("Target" + "\n" + str(pair[1]) + "\n")
            outliers_file.write(targetsequences_max[pair[1]] + "\n")

            outliers_file.write("Predicted Similarity: " + str(all_similarities[idx]) + "\n")
            outliers_file.write("E-value: " + str(all_e_values[idx]) + "\n")

        loge = np.ma.masked_invalid(np.log10(all_e_values))
        idxs = np.where(loge < -250)[0]

    with open("inliers.txt", "w", encoding="utf-8") as inliers_file:
        for idx in idxs:
            pair = all_targets[idx]
            inliers_file.write("Query" + "\n" + str(pair[0]) + "\n")
            inliers_file.write(querysequences_max[pair[0]] + "\n")
            inliers_file.write("Target" + "\n" + str(pair[1]) + "\n")
            inliers_file.write(targetsequences_max[pair[1]] + "\n")

            inliers_file.write("Predicted Similarity: " + str(all_similarities[idx]) + "\n")
            inliers_file.write("E-value: " + str(all_e_values[idx]) + "\n")


def get_data(hits_path: str):
    """Parses the outputted results and aggregates everything
    into lists and dictionaries"""
    similarity_hits_dict = {}
    all_similarities = []

    all_e_values = []
    all_biases = []
    all_targets = []
    for queryhits in os.listdir(hits_path):
        queryname = queryhits.strip(".txt")
        with open(f"{hits_path}/{queryhits}", "r", encoding="utf-8") as similarities:
            similarities = similarities.readlines()
            if queryname not in all_hits_max.keys():
                continue
            similarity_hits_dict[queryname] = {}

            for line in similarities:
                if "Distance" in line:
                    continue
                try:
                    target = line.split()[0].strip("\n")
                    if target not in all_hits_max[queryname].keys():
                        continue
                    similarity = float(line.split()[1].strip("\n"))
                    all_similarities.append(similarity)

                    similarity_hits_dict[queryname][target] = similarity

                    all_e_values.append(all_hits_max[queryname][target][0])
                    all_biases.append(all_hits_max[queryname][target][2])
                    all_targets.append((queryname, target))
                except ValueError as exception:
                    print(exception)
                    continue

    np.save("all_similarities", np.array(all_similarities), allow_pickle=True)
    np.save("all_biases", np.array(all_biases), allow_pickle=True)
    np.save("all_e_values", np.array(all_e_values), allow_pickle=True)

    numhits = np.sum([len(similarity_hits_dict[q]) for q in list(similarity_hits_dict.keys())])
    print(f"Got {numhits} total hits from our model")
    return similarity_hits_dict, all_similarities, all_e_values, all_biases, numhits


if __name__ == "__main__":
    QUERY_FILENUM = 4

    querysequences_max, targetsequences_max, all_hits_max = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=QUERY_FILENUM,
    )

    MODEL_RESULTS_PATH = (
        "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities"
    )

    TEMP_DATA_FILE = "/xdisk/twheeler/daphnedemekas/data1_distance_sum_hits.txt"

    # similarity_hits_dict, all_similarities, all_e_values, all_biases, _ = get_data(hits_path)

    generate_roc(TEMP_DATA_FILE, MODEL_RESULTS_PATH, all_hits_max, "ResNet1d/blosum_eval/roc.png")

    # all_similarities = np.load("all_similarities.npy")
    # all_e_values = np.load("all_e_values.npy")
    # all_biases = np.load("all_biases.npy")
