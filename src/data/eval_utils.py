"""File with functions to evaluate model """
import os
from typing import Tuple

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pickle
import pdb
from src.data.hmmerhits import FastaFile


COLORS = ["r", "c", "g", "k"]


def update(d1, d2):
    c = d1.copy()
    for key in d2:
        if key in d1:
            c[key].update(d2[key])
        else:
            c[key] = d2[key]
    return c


def get_evaluation_data(
    queryfile="data/evaluationqueries.fa",
    save_dir=None,
    targethitsfile="/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargetdict.pkl",
    evaltargetfastafile="data/evaluationtargets.fa",
) -> Tuple[dict, dict, dict]:
    """Taking advantage of our current data structure of nested directories
    holding fasta files to quickly get all hmmer hits and sequence dicts for all
    queries in the input query id file and all target sequences in all of num_files"""

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    queryfasta = FastaFile(queryfile)
    querysequences = queryfasta.data
    print(f"Number of query sequences: {len(querysequences)}")
    filtered_query_sequences = {}

    targetsequencefasta = FastaFile(evaltargetfastafile)
    targetsequences = targetsequencefasta.data
    print(f"Number of target sequences: {len(targetsequences)}")

    if save_dir is not None:  # only return those that we don't already have

        existing_queries = [f.strip(".txt") for f in os.listdir(save_dir)]

        print(f"Cleaning out {len(existing_queries)} queries that we already have in results...")
        for query, value in querysequences.items():
            if query not in existing_queries:
                filtered_query_sequences.update({query: value})

        querysequences = filtered_query_sequences

    with open(targethitsfile, "rb") as file:
        all_target_hits = pickle.load(file)

    print(f"Number of target HMMER hits: {len(all_target_hits)}")

    return querysequences, targetsequences, all_target_hits


def plot_mean_e_values(
    distance_list: list,
    e_values_list: list,
    biases: list,
    min_threshold: int = 0,
    max_threshold: int = 300,
    outputfilename: str = "evaluemeans",
    plot_stds: bool = True,
    _plot_lengths: bool = False,
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
    biases = np.array(biases)

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
    if _plot_lengths:
        plt.fill_between(
            thresholds,
            np.array(means) - np.array(lengths),
            np.array(means) + np.array(lengths),
            alpha=0.5,
            color="orange",
        )
    plt.title(title)
    plt.ylim(-20, 0)
    # plt.xlim(0,100)
    plt.ylabel("Log E value means")
    plt.xlabel("Similarity Threshold")
    plt.savefig(f"ResNet1d/results/{outputfilename}.png")


def plot_lengths(distance_list1, distance_list2, distance_list3):
    thresholds = np.linspace(
        0, max(np.max(distance_list1), np.max(distance_list2), np.max(distance_list3)), 100
    )
    all_distance1 = np.array(distance_list1)
    all_distance2 = np.array(distance_list2)
    all_distance3 = np.array(distance_list3)
    lengths1 = []
    lengths2 = []
    lengths3 = []

    for threshold in thresholds:
        idx1 = np.where(all_distance1 > threshold)[0]
        idx2 = np.where(all_distance2 > threshold)[0]
        idx3 = np.where(all_distance3 > threshold)[0]
        lengths1.append(len(idx1))
        lengths2.append(len(idx2))
        lengths3.append(len(idx3))
    plt.plot(thresholds, lengths1, label="Alignment Model")
    plt.plot(thresholds, lengths2, label="Blosum Model")
    plt.plot(thresholds, lengths3, label="K-mer model")
    plt.xlabel("Similarity threshold")
    plt.ylabel("Number of hits above threshold")
    plt.title("Number of hits at different similarity thresholds across models")
    plt.legend()
    plt.savefig("ResNet1d/eval/lengths.png")


def plot_roc_curve(
    figure_path: str,
    numpos_per_evalue: list,
    numhits: int,
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
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
    print("Generating ROC plot")

    num_thresholds = len(evalue_thresholds)
    _, axis = plt.subplots(figsize=(10, 10))
    axis.set_title(f"{os.path.splitext(os.path.basename(figure_path))[0]}")

    num_positives = [0] * num_thresholds
    num_decoys = [0] * num_thresholds

    filtrations = []
    recalls = []

    recall = None
    filtration = None
    datafile = open(filename, "r")

    for idx, line in tqdm.tqdm(enumerate(datafile)):
        line = line.split()
        classnames = [line[3 + i] for i in range(num_thresholds)]
        for i in range(num_thresholds):
            if classnames[i] == "P":
                num_positives[i] += 1
            elif classnames[i] == "D":
                num_decoys[i] += 1
        filtration = [num_decoys[i] / numhits for i in range(num_thresholds)]

        if 100 * filtration[0] > 25:
            datafile.close()
            for i in range(num_thresholds):
                axis.plot(
                    np.array(filtrations)[:, i],
                    np.array(recalls)[:, i],
                    f"{COLORS[i]}--",
                    linewidth=2,
                    label=evalue_thresholds[i],
                )

            # axis.plot([75, 100], [25, 0], "k--", linewidth=2)

            axis.set_xlabel("filtration")
            axis.set_ylabel("recall")
            plt.ylim(0, 100)
            plt.legend()
            plt.savefig(f"{figure_path}", bbox_inches="tight")
            plt.close()
            return None

        if idx % 50000 == 0 and (100 * (1 - filtration[0]) > 75):
            recall = [num_positives[i] / numpos_per_evalue[i] for i in range(num_thresholds)]
            filtrations.append([100 * (1 - filtration[i]) for i in range(num_thresholds)])
            recalls.append([100 * recall[i] for i in range(num_thresholds)])

    datafile.close()
    for i in range(num_thresholds):
        axis.plot(
            np.array(filtrations)[:, i],
            np.array(recalls)[:, i],
            f"{COLORS[i]}--",
            linewidth=2,
            label=evalue_thresholds[i],
        )

    # axis.plot([75, 100], [25, 0], "k--", linewidth=2)

    axis.set_xlabel("filtration")
    axis.set_ylabel("recall")
    plt.legend()
    plt.savefig(f"{figure_path}_FULL.png", bbox_inches="tight")
    plt.close()


def get_sorted_pairs(modelhitsfile: str) -> Tuple[list, list]:
    """parses the output file from our model
    and returns a list of scores and query-target
    pairs for the results that are also in hmmer hits"""
    all_scores = []
    all_pairs = []

    print(f"Iterating over {modelhitsfile}..")
    for queryhits in tqdm.tqdm(os.listdir(modelhitsfile)):
        queryname = queryhits.strip(".txt")

        with open(f"{modelhitsfile}/{queryhits}", "r") as file:

            for line in file:
                if "Distance" in line:
                    continue
                target = line.split()[0].strip("\n")

                score = float(line.split()[1].strip("\n"))
                all_scores.append(score)

                all_pairs.append((queryname, target))

    sortedidx = np.argsort(all_scores)[::-1]
    del all_scores
    sorted_pairs = [all_pairs[i] for i in sortedidx]
    del all_pairs

    return sorted_pairs


def write_datafile(
    pairs: list,
    hmmerhits: dict,
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
    filename: str = "data.txt",
) -> list:
    """This opens a data file and counts the recall as it descends through the results
    Returns the number of positives for 4 evalue thresholds
    TODO: This code can be improved /generalized"""
    numpos_per_evalue = [0, 0, 0, 0]
    print("Writing to file...")
    numhits = 0
    with open(filename, "w", encoding="utf-8") as datafile:
        for pair in tqdm.tqdm(pairs):
            query, target = pair[0], pair[1]
            try:
                evalue = None
                if query not in hmmerhits or target not in hmmerhits[query]:
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
            except Exception as e:
                print(e)
                pdb.set_trace()

            datafile.write(
                f"{query}          {target}          {evalue}          \
                    {classname1}          {classname2}          \
                        {classname3}          {classname4}"
                + "\n"
            )
            numhits += 1

    return numpos_per_evalue, numhits


def generate_roc(modelhitsfile, hmmerhits, figure_path, temp_file):
    sorted_pairs = get_sorted_pairs(modelhitsfile)
    print(f"Length of sorted pairs: {len(sorted_pairs)}")

    numpos_per_evalue, numhits = write_datafile(
        sorted_pairs, hmmerhits, evalue_thresholds=[1e-10, 1e-4, 1e-1, 10], filename=temp_file
    )
    print("Wrote files")
    print(f"Num pos per evalue: {numpos_per_evalue}")
    print(f"Num hits: {numhits}")
    plot_roc_curve(figure_path, numpos_per_evalue, numhits, filename=temp_file)


def get_outliers(
    all_similarities: list,
    all_e_values: list,
    all_targets: list,
    querysequences_max,
    targetsequences_max,
):
    """Writes outliers to a text file
    where outliers are defined as our hits that have large similairty
    but high e value"""
    d_idxs = np.where(all_similarities > 150)[0]

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


def get_data(hits_path: str, all_hits_max: dict, savedir=None):
    """Parses the outputted results and aggregates everything
    into lists and dictionaries"""

    similarity_hits_dict = {}
    all_similarities = []

    all_e_values = []
    all_biases = []
    all_targets = []
    print(hits_path)
    for queryhits in tqdm.tqdm(os.listdir(hits_path)):
        queryname = queryhits.strip(".txt")
        with open(f"{hits_path}/{queryhits}", "r") as similarities:
            if queryname not in all_hits_max:
                print(queryname)

                continue
            similarity_hits_dict[queryname] = {}

            for line in similarities:
                if "Distance" in line:
                    continue
                target = line.split()[0].strip("\n")
                if target not in all_hits_max[queryname]:
                    print(target)
                    continue

                similarity = float(line.split()[1].strip("\n"))
                all_similarities.append(similarity)

                similarity_hits_dict[queryname][target] = similarity

                all_e_values.append(all_hits_max[queryname][target][0])
                all_biases.append(all_hits_max[queryname][target][2])
                all_targets.append((queryname, target))

    if savedir is not None:
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        print(f"Saving to {savedir}:")
        np.save(f"{savedir}/all_similarities", np.array(all_similarities), allow_pickle=True)
        np.save(f"{savedir}/all_biases", np.array(all_biases), allow_pickle=True)
        np.save(f"{savedir}/all_e_values", np.array(all_e_values), allow_pickle=True)
        np.save(f"{savedir}/all_targets", np.array(all_targets), allow_pickle=True)

        with open(f"{savedir}/hits_dict.pkl", "wb") as file:
            pickle.dump(similarity_hits_dict, file)

    numhits = np.sum([len(similarity_hits_dict[q]) for q in list(similarity_hits_dict.keys())])
    print(f"Got {numhits} total hits from our model")
    return (
        similarity_hits_dict,
        all_similarities,
        all_e_values,
        all_biases,
        numhits,
    )
