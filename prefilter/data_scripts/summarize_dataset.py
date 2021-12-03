import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


def summarize_dataset(json_files, savefig=None):
    lengths = []
    sequence_to_count = defaultdict(int)
    labels_per_sequence = []

    family_to_count = defaultdict(int)
    for f in json_files:
        with open(f, "r") as src:
            sequence_to_labels = json.load(src)
        for sequence, label_list in sequence_to_labels.items():
            for label in label_list:
                family_to_count[label] += 1

        sequences = list(sequence_to_labels.keys())
        labels = list(sequence_to_labels.values())
        l = list(map(len, sequences))
        lengths += l
        l = list(map(len, labels))
        labels_per_sequence += l

    s = "mean length: {:.3f}, median: {:.3f}, 25th percentile: {:.3f}\n 75th"
    s += " percentile {:.3f}, number of seq. {:d}"

    s = s.format(
        np.mean(lengths),
        np.median(lengths),
        np.percentile(lengths, 25),
        np.percentile(lengths, 75),
        len(lengths),
    )

    spf = "mean members: {:.3f}, median: {:.3f}, 25th percentile: {:.3f}\n 75th"
    spf += " percentile {:.3f}, number of families. {:d}\n"
    spf += " min members: {:.3f}, max members: {:.3f} "
    x = list(family_to_count.values())
    spf = spf.format(
        np.mean(x),
        np.median(x),
        np.percentile(x, 25),
        np.percentile(x, 75),
        len(family_to_count),
        np.min(x),
        np.max(x),
    )

    if savefig is not None:
        fig, ax = plt.subplots(ncols=3, figsize=(13, 10))

        ax[0].set_title(s, fontsize=8)
        ax[0].hist(lengths, bins=100, histtype="step")
        ax[0].set_xlabel("sequence length")

        ax[1].set_title("number of labels per sequence", fontsize=8)
        ax[1].hist(labels_per_sequence, bins=100, histtype="step")
        ax[1].set_xlabel("num labels")

        ax[2].set_title(spf, fontsize=8)
        ax[2].hist(list(family_to_count.values()), bins=50, histtype="step")
        ax[1].set_xlabel("members in family")
        plt.savefig(savefig)
        plt.close()


def parser():
    ap = ArgumentParser()
    ap.add_argument(
        "--directory", required=True, type=str, help="where json files are stored"
    )
    ap.add_argument(
        "--glob_string", required=True, type=str, help="which pattern you want to match"
    )
    ap.add_argument("--save_fig", type=str)
    return ap.parse_args()


if __name__ == "__main__":
    args = parser()

    directory = args.directory
    glob_string = args.glob_string
    save_path = args.save_fig
    json_files = glob(os.path.join(directory, glob_string))
    if not len(json_files):
        raise ValueError(
            "glob string {}j does not contain any json files".format(
                os.path.join(directory, glob_string)
            )
        )
    summarize_dataset(json_files, savefig=save_path)
