"""File with functions to evaluate model """
import os
from typing import Tuple

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pickle
from src.analysis.plotting import *


def get_filtration_recall(
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
    filename: str = "data.txt",
):
    if "max" in filename:
        print("max numpos")
        numpos_per_evalue = [355203, 598800, 901348, 3607355]
        alldecoys = [2342448072, 2342448072, 2342448072, 2342448072]
    else:
        print("normal numpos")
        numpos_per_evalue = [354984, 593354, 839953, 886633]
        alldecoys = [2345180299, 2345180299, 2345180299, 2345180299]
    print("Getting Filtration & Recall")

    num_thresholds = len(evalue_thresholds)

    num_positives = [0] * num_thresholds
    num_decoys = [0] * num_thresholds

    filtrations = []
    recalls = []

    recall = None
    filtration = None
    print(f"Reading file {filename}")
    datafile = open(filename, "r")

    for idx, line in tqdm.tqdm(enumerate(datafile)):
        line = line.split()
        classnames = [line[3 + i] for i in range(num_thresholds)]
        for i in range(num_thresholds):
            if classnames[i] == "P":
                num_positives[i] += 1
            elif classnames[i] == "D":
                num_decoys[i] += 1
        filtration = [num_decoys[i] / alldecoys[i] for i in range(num_thresholds)]

        if idx % 1000 == 0 and (100 * (1 - filtration[0]) > 75):
            recall = [num_positives[i] / numpos_per_evalue[i] for i in range(num_thresholds)]

            filtrations.append([100 * (1 - filtration[i]) for i in range(num_thresholds)])
            recalls.append([100 * recall[i] for i in range(num_thresholds)])

        elif 100 * (1 - filtration[0]) < 75:
            datafile.close()
            return filtrations, recalls
    return filtrations, recalls


def get_sorted_pairs(all_scores, all_pairs) -> Tuple[list, list]:
    """parses the output file from our model
    and returns a list of scores and query-target
    pairs for the results that are also in hmmer hits"""
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
    print(f"Writing to file {filename}..")

    with open(filename, "w", encoding="utf-8") as datafile:
        for pair in tqdm.tqdm(pairs):
            query, target = pair[0], pair[1]

            evalue = None
            if query not in hmmerhits or target not in hmmerhits[query]:
                classname1 = classname2 = classname3 = classname4 = "D"
            else:
                evalue = hmmerhits[query][target][0]
                classname1 = classname2 = classname3 = classname4 = "M"

                if evalue < evalue_thresholds[3]:
                    classname4 = "P"
                    if evalue < evalue_thresholds[2]:
                        classname3 = "P"
                        if evalue < evalue_thresholds[1]:
                            classname2 = "P"
                            if evalue < evalue_thresholds[0]:
                                classname1 = "P"
            datafile.write(
                f"{query}          {target}          {evalue}          \
                    {classname1}          {classname2}          \
                        {classname3}          {classname4}"
                + "\n"
            )


def get_roc_data(hmmer_hits_dict: dict, temp_file: str, sorted_pairs=None, **kwargs):
    if os.path.exists(f"{temp_file}_filtration.pickle"):
        with open(f"{temp_file}_filtration.pickle", "rb") as pickle_file:
            filtrations = pickle.load(pickle_file)
        with open(f"{temp_file}_recall.pickle", "rb") as pickle_file:
            recalls = pickle.load(pickle_file)
        return filtrations, recalls

    if not os.path.exists(temp_file):
        write_datafile(
            sorted_pairs,
            hmmer_hits_dict,
            evalue_thresholds=[1e-10, 1e-4, 1e-1, 10],
            filename=temp_file,
        )
    print("Wrote files")
    filtrations, recalls = get_filtration_recall(filename=temp_file)

    print(f"Saving filtrations and recalls to {temp_file}_filtration and {temp_file}_recall")

    with open(f"{temp_file}_filtration.pickle", "wb") as filtrationfile:
        pickle.dump(filtrations, filtrationfile)
    with open(f"{temp_file}_recall.pickle", "wb") as recallfile:
        pickle.dump(recalls, recallfile)

    return filtrations, recalls


def generate_roc(figure_path: str, hmmerhits: dict, filename: str, sorted_pairs):
    """Pipeline to write data to file and generate the ROC plot
    This will then delete the file as well as its massive and not useful"""
    filtrations, recalls = get_roc_data(hmmerhits, filename, sorted_pairs)
    plot_roc_curve(figure_path, filtrations, recalls)


def get_data(
    model_results_path: str,
    hmmer_hits_dict: dict,
    data_savedir=None,
    plot_roc=True,
    **kwargs,
):
    """Parses the outputted results and aggregates everything
    into lists and dictionaries"""

    if (data_savedir is not None and plot_roc is False) or os.path.exists(
        f"{data_savedir}/sorted_pairs.npy"
    ):
        print(f"Getting saved data from {data_savedir}")
        all_similarities = np.load(f"{data_savedir}/similarities.npy")
        all_e_values = np.load(f"{data_savedir}/all_e_values.npy")
        all_biases = np.load(f"{data_savedir}/all_biases.npy")
        if plot_roc is True:
            sorted_pairs = np.load(f"{data_savedir}/sorted_pairs.npy")
        else:
            sorted_pairs = None
        return (
            all_similarities,
            all_e_values,
            all_biases,
            sorted_pairs,
        )

    similarities = []

    all_e_values = []
    all_biases = []
    all_targets = []
    all_scores = []

    print(model_results_path)

    if "CPU" in model_results_path:
        nprobe = model_results_path.split("/")[-1].split("-")[-1]
        reversed_path = f"/xdisk/twheeler/daphnedemekas/prefilter-output/reversed-{nprobe}"
    else:
        reversed_path = model_results_path + "-reversed"
    print(f"Reversed path :{reversed_path}")

    for queryhits in tqdm.tqdm(os.listdir(model_results_path)):
        queryname = queryhits.strip(".txt")
        # get positives
        with open(f"{model_results_path}/{queryhits}", "r") as file:
            for line in file:
                if "Distance" in line:
                    continue
                target = line.split()[0].strip("\n").strip(".pt")
                similarity = float(line.split()[1].strip("\n"))
                # if there is a decoy, then collect targets from reversed results
                if queryname not in hmmer_hits_dict or target not in hmmer_hits_dict[queryname]:
                    continue

                all_targets.append((queryname, target))
                all_scores.append(similarity)
                all_e_values.append(hmmer_hits_dict[queryname][target][0])
                all_biases.append(hmmer_hits_dict[queryname][target][2])
                similarities.append(similarity)

        # get decoys
        if os.path.exists(f"{reversed_path}/{queryhits}"):
            with open(f"{reversed_path}/{queryhits}", "r") as file:
                for line in file:
                    if "Distance" in line:
                        continue
                    target = line.split()[0].strip("\n")

                    if queryname not in hmmer_hits_dict or target not in hmmer_hits_dict[queryname]:
                        similarity = float(line.split()[1].strip("\n"))
                        all_targets.append((queryname, target))
                        all_scores.append(similarity)

    assert len(all_scores) == len(all_targets)
    print("Sorting pairs...")
    sorted_pairs = get_sorted_pairs(all_scores, all_targets)
    if data_savedir is not None:
        if not os.path.exists(data_savedir):
            os.mkdir(data_savedir)
        print(f"Saving to {data_savedir}:")
        np.save(
            f"{data_savedir}/similarities",
            np.array(similarities, dtype="half"),
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
        # np.save(f"{data_savedir}/sorted_pairs", np.array(sorted_pairs), allow_pickle=True)

    return (similarities, all_e_values, all_biases, sorted_pairs)


def get_similarity(pair):
    return pair[-1]


def get_data_for_roc(
    model_results_path: str,
    hmmer_hits_dict: dict,
    data_savedir=None,
    plot_roc=True,
    **kwargs,
):
    """Parses the outputted results and aggregates everything
    into lists and dictionaries"""

    if (data_savedir is not None and plot_roc is False) or os.path.exists(
        f"{data_savedir}/sorted_pairs.npy"
    ):
        print(f"Getting saved data from {data_savedir}")
        if plot_roc is True:
            sorted_pairs = np.load(f"{data_savedir}/sorted_pairs.npy")
        else:
            sorted_pairs = None
        return (sorted_pairs,)

    all_pairs = []

    print(model_results_path)

    reversed_path = model_results_path + "-reversed"
    print(f"Reversed path :{reversed_path}")

    for queryhits in tqdm.tqdm(os.listdir(model_results_path)):
        queryname = queryhits.strip(".txt")
        # get positives
        with open(f"{model_results_path}/{queryhits}", "r") as file:
            for line in file:
                if "Distance" in line:
                    continue
                target = line.split()[0].strip("\n").strip(".pt")
                # if there is a decoy, then collect targets from reversed results
                if queryname not in hmmer_hits_dict or target not in hmmer_hits_dict[queryname]:
                    continue

                similarity = float(line.split()[1].strip("\n"))
                all_pairs.append((queryname, target, similarity))

        if os.path.exists(f"{reversed_path}/{queryhits}"):
            with open(f"{reversed_path}/{queryhits}", "r") as file:
                for line in file:
                    if "Distance" in line:
                        continue
                    target = line.split()[0].strip("\n")

                    if queryname not in hmmer_hits_dict or target not in hmmer_hits_dict[queryname]:
                        similarity = float(line.split()[1].strip("\n"))
                        all_pairs.append((queryname, target, similarity))

    print("Sorting pairs...")

    all_pairs.sort(key=get_similarity, reverse=True)

    return all_pairs