"""File with functions to evaluate model """
import os
from typing import Tuple

import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pickle
import pdb

COLORS = ["mediumseagreen", "darkblue", "mediumvioletred", "darkorchid", "tomato"]


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
    scatter_size: int = 3,
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
    # plt.title(title + ", Full")
    # plt.ylabel("Log E value means")
    # plt.xlabel("Similarity Threshold")
    # plt.savefig(f"ResNet1d/results/{outputfilename}-full.png")

    plt.title(title)
    plt.ylim(-20, 0)
    plt.xlim(0,100)
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
    filtrations: list,
    recalls: list,
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
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

    for i in range(num_thresholds):
        axis.plot(
            np.array(filtrations)[:, i],
            np.array(recalls)[:, i],
            f"{COLORS[i]}",
            linewidth=2,
            label=evalue_thresholds[i],
        )
    axis.set_xlabel("filtration")
    axis.set_ylabel("recall")
    axis.ticklabel_format(style='plain', axis='x', useOffset=False)
    axis.grid()
    axis.set_ylim(0, 100)
    plt.legend()
    plt.savefig(f"{figure_path}", bbox_inches="tight")
    plt.close()


def get_filtration_recall(
    evalue_thresholds: list = [1e-10, 1e-4, 1e-1, 10],
    filename: str = "data.txt",
):
    if 'max' in filename:
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
    #numpos_per_evalue = [0, 0, 0, 0]
    print(f"Writing to file {filename}..")
    #numhits = 0
    #numdecoys = 0

    with open(filename, "w", encoding="utf-8") as datafile:
        for pair in tqdm.tqdm(pairs):
            query, target = pair[0], pair[1]

            evalue = None
            if query not in hmmerhits or target not in hmmerhits[query]:
                classname1 = classname2 = classname3 = classname4 = "D"
                #numdecoys += 1
            else:
                evalue = hmmerhits[query][target][0]
                classname1 = classname2 = classname3 = classname4 = "M"

                if evalue < evalue_thresholds[3]:
                    classname4 = "P"
                    #numpos_per_evalue[3] += 1
                    if evalue < evalue_thresholds[2]:
                        classname3 = "P"
                        #numpos_per_evalue[2] += 1
                        if evalue < evalue_thresholds[1]:
                            classname2 = "P"
                            #numpos_per_evalue[1] += 1
                            if evalue < evalue_thresholds[0]:
                                classname1 = "P"
                                #numpos_per_evalue[0] += 1
            datafile.write(
                f"{query}          {target}          {evalue}          \
                    {classname1}          {classname2}          \
                        {classname3}          {classname4}"
                + "\n"
            )
            #numhits += 1

    # print(f"Numpos per evalue: {numpos_per_evalue}")
    # print(f"Number of hits: {numhits}")
    # print(f"Number of decoys: {numdecoys}")

    #return numpos_per_evalue, numdecoys


def get_roc_data(hmmer_hits_dict: dict, temp_file: str, sorted_pairs = None, **kwargs):

    if os.path.exists(f"{temp_file}_filtration.pickle"):
        with open(f"{temp_file}_filtration.pickle", 'rb') as pickle_file:
            filtrations = pickle.load(pickle_file)
        with open(f"{temp_file}_recall.pickle", 'rb') as pickle_file:
            recalls = pickle.load(pickle_file)
        return filtrations, recalls

    write_datafile(
        sorted_pairs, hmmer_hits_dict, evalue_thresholds=[1e-10, 1e-4, 1e-1, 10], filename=temp_file
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



def get_data(model_results_path: str, hmmer_hits_dict: dict, data_savedir=None, **kwargs):
    """Parses the outputted results and aggregates everything
    into lists and dictionaries"""

    if data_savedir is not None and os.path.exists(f"{data_savedir}/sorted_pairs.npy"):
        print(f"Getting saved data from {data_savedir}")
        all_similarities = np.load(f"{data_savedir}/similarities.npy")
        all_e_values = np.load(f"{data_savedir}/all_e_values.npy")
        all_biases = np.load(f"{data_savedir}/all_biases.npy")
        sorted_pairs = np.load(f"{data_savedir}/sorted_pairs.npy")
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
    

    #with open ("/xdisk/twheeler/daphnedemekas/prefilter/target_names.txt", "r") as f:
    #    target_names = [t.strip("\n") for  t in f.readlines()]
    print(model_results_path)

    if 'CPU' in model_results_path:
        nprobe = model_results_path.split("/")[-1].split("-")[-1]
        reversed_path = f"/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/reversed-{nprobe}" 
    else:
        reversed_path = model_results_path + "-reversed"
    print(f"Revered path :{reversed_path}")
    
    #TODO: here get a dictionary query: target: score for the results from shuffeld data not in hmmer
    reversed_results = {}
    for queryhits in tqdm.tqdm(os.listdir(model_results_path)):
        queryname = queryhits.strip(".txt")
        with open(f"{model_results_path}/{queryhits}", "r") as file:

            for line in file:
                if "Distance" in line:
                    continue
                target = line.split()[0].strip("\n").strip(".pt")
       #         if target not in target_names:
       #             continue
                #all_targets.append((queryname, target))
                similarity = float(line.split()[1].strip("\n"))
                #if there is a decoy, then collect targets from reversed results
                if queryname not in hmmer_hits_dict or target not in hmmer_hits_dict[queryname]:
                    if os.path.exists(f"{reversed_path}/{queryhits}"):
                        if queryname in reversed_results:
                            if target in reversed_results[queryname]:
                                #print("getting reversed similarity")
                                similarity = reversed_results[queryname][target]
                            else:
                                continue
                        else:
                            reversed_results[queryname] = {}
                            with open(f"{reversed_path}/{queryhits}") as file:
                                for line in file:
                                    if "Distance" in line:
                                        continue
                                    target = line.split()[0].strip("\n")
                                    s = float(line.split()[1].strip("\n"))
                                    reversed_results[queryname][target] = s
                            similarity = reversed_results[queryname][target]
                else:
                    similarities.append(similarity)

                    all_e_values.append(hmmer_hits_dict[queryname][target][0])
                    all_biases.append(hmmer_hits_dict[queryname][target][2])

                all_targets.append((queryname, target))
                all_scores.append(similarity)
    assert len(all_scores) == len(all_targets)
    print("Sorting pairs...")
    sorted_pairs = get_sorted_pairs(all_scores, all_targets)
    if data_savedir is not None:
        if not os.path.exists(data_savedir):
            os.mkdir(data_savedir)
        print(f"Saving to {data_savedir}:")
        np.save(f"{data_savedir}/similarities", np.array(similarities,dtype='half'), allow_pickle=True)
        np.save(f"{data_savedir}/all_biases", np.array(all_biases,dtype='half'), allow_pickle=True)
        np.save(f"{data_savedir}/all_e_values", np.array(all_e_values,dtype='half'), allow_pickle=True)
        #np.save(f"{data_savedir}/sorted_pairs", np.array(sorted_pairs), allow_pickle=True)

    return (similarities, all_e_values, all_biases, sorted_pairs)
