import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.utils import (fasta_from_file, handle_figure_path,
                       load_sequences_and_labels, parse_labels)


def histogram_number_of_seqs_per_family(fasta_files, savefig=None):
    """
    Creates and saves a histogram of the number of sequences in the list of fasta files.
    Useful for investigating the imbalance present in a dataset.
    :param savefig: where to save the data
    :type savefig: str
    :param fasta_files: files to anaylze
    :type fasta_files: str
    :return: None
    :rtype: None
    """
    label_to_sequence_count = defaultdict(int)
    for fasta in fasta_files:
        labelset, sequences = fasta_from_file(fasta)
        # parse labels, get
        for labelstring, sequence in zip(labelset, sequences):
            labels = parse_labels(labelstring)
            if labels is None:
                print(labelstring)
                continue
            else:
                for pfam_accession_id in labels:
                    label_to_sequence_count[pfam_accession_id] += 1

    fig, ax = plt.subplots()
    for i, (label, count) in enumerate(label_to_sequence_count.items()):
        ax.bar(i, np.log(count))

    ax.set_xticklabels(list(label_to_sequence_count.keys()))

    plt.savefig(savefig)
    plt.close()


def neighborhood_labels(fasta_files: List[str], figure_path: str) -> None:
    """
    % of all sequences with multiple labels
    % of seqs in each family with multiple labels
    :param figure_path: where to save the figure
    :type figure_path: str
    :param fasta_files: list of fasta files to analyze
    :type fasta_files: List[str]
    :return: None
    :rtype: None
    """
    labels_and_sequence = load_sequences_and_labels(fasta_files)

    family_to_multilabel_membership = {}
    family_to_n_seq_per_family = defaultdict(int)

    for labelset, sequence in labels_and_sequence:
        primary_label = labelset[0]
        family_to_n_seq_per_family[primary_label] += 1
        multi_label = 1 if len(labelset) > 1 else 0
        if primary_label not in family_to_multilabel_membership:
            family_to_multilabel_membership[primary_label] = [1, multi_label]
        else:
            family_to_multilabel_membership[primary_label][0] += 1
            family_to_multilabel_membership[primary_label][1] += multi_label

    n_seq_per_family_to_family = defaultdict(list)
    for family, n_seq in family_to_n_seq_per_family.items():
        n_seq_per_family_to_family[n_seq].append(family)
    n_seq_per_family_to_total_number_sequences = defaultdict(int)
    n_seq_per_family_to_number_multilabel = defaultdict(int)

    for n_seq, family_list in n_seq_per_family_to_family.items():
        for family in family_list:
            n_seq_per_family_to_number_multilabel[n_seq] += family_to_multilabel_membership[family][
                1
            ]
            n_seq_per_family_to_total_number_sequences[n_seq] += family_to_multilabel_membership[
                family
            ][0]

    sorted_n_seqs = sorted(list(n_seq_per_family_to_family.keys()))
    family_membership = []
    multi_label_membership = []
    for n_seq in sorted_n_seqs:
        multi_label_membership.append(n_seq_per_family_to_number_multilabel[n_seq])
        family_membership.append(n_seq_per_family_to_total_number_sequences[n_seq])

    fig, ax = plt.subplots(nrows=3, figsize=(13, 10))
    first_bar = np.array([x[0] for x in family_to_multilabel_membership.values()])
    second_bar = np.array([x[1] for x in family_to_multilabel_membership.values()])
    percent_bar = np.array([x[1] / x[0] for x in family_to_multilabel_membership.values()])

    idx = np.argsort(first_bar)
    first_bar = first_bar[idx]
    second_bar = second_bar[idx]
    percent_bar = percent_bar[idx]

    ax[0].bar(np.arange(len(percent_bar)), percent_bar)
    ax[0].set_title(
        "percent of sequences in family with neighborhood labels (sorted by membership)"
    )

    ax[1].bar(
        np.arange(len(first_bar)),
        np.log(first_bar),
        label="num sequences in family",
    )
    ax[1].bar(
        np.arange(len(second_bar)),
        np.log(second_bar),
        label="num sequences in family with multiple labels",
    )
    ax[1].set_title("sequences with and without multiple labels per family (log scale)")
    ax[1].set_xlabel("unique family (sorted by membership)")
    ax[1].legend()

    ratio = np.array(multi_label_membership) / np.array(family_membership)
    ax[2].bar(sorted_n_seqs, ratio)
    ax[2].set_xlabel("percent of sequences in families with N sequences that have multiple labels")

    plt.savefig(handle_figure_path(figure_path))
    plt.close()


def n_seq_per_family(seq_and_labels: List[Tuple[List[str], str]]) -> Dict[str, int]:

    family_to_n_seq_per_family = defaultdict(int)
    for labelset, sequence in seq_and_labels:
        for label in labelset:
            family_to_n_seq_per_family[label] += 1

    return family_to_n_seq_per_family


def get_neighborhood_labels(seq_and_labels):
    labels = []
    for labelset, sequence in seq_and_labels:
        if len(labelset) > 1:
            for label in labelset[1:]:
                labels.append(label)
    return labels


def compare_valid_and_train_labels(hparams: dict, figure_path: str) -> None:

    val_files = hparams["val_files"]
    train_files = hparams["train_files"]

    val_seq_and_labels = load_sequences_and_labels(val_files)
    train_seq_and_labels = load_sequences_and_labels(train_files)
    train_family_to_membership = n_seq_per_family(train_seq_and_labels)
    val_family_to_membership = n_seq_per_family(val_seq_and_labels)

    val_labels = get_neighborhood_labels(val_seq_and_labels)
    train_labels = [t for v in train_seq_and_labels for t in v[0]]

    set_train_labels = set(train_labels)
    set_val_labels = set(val_labels)
    shared = 0
    unique = 0
    for v in set_val_labels:
        if v in set_train_labels:
            shared += 1
        else:
            unique += 1

    print(
        f"number train labels: {len(set_train_labels)}, num. val labels: {len(set_val_labels)}, number of labels in val that are in train: {shared}, "
        f"number of labels in val that are not in train: {unique}"
    )

    num_train_labels = []
    num_val_labels = []

    for unique_validation_label in set_val_labels:
        # this will FAIL if unique_validation_label is not present in train
        num_train_labels.append(train_family_to_membership[unique_validation_label])
        num_val_labels.append(val_family_to_membership[unique_validation_label])

    num_train_labels = np.array(num_train_labels)
    num_val_labels = np.array(num_val_labels)
    idx = np.argsort(num_train_labels)
    num_train_labels = num_train_labels[idx]
    num_val_labels = num_val_labels[idx]
    fig, ax = plt.subplots(nrows=3, figsize=(13, 10))

    ax[0].bar(
        np.arange(len(num_train_labels)),
        num_train_labels,
        label="train labels",
    )
    ax[0].bar(np.arange(len(num_val_labels)), num_val_labels, label="val labels")
    ax[0].set_title("number of training and validation labels per family")
    ax0 = ax[0].twinx()
    ax0.plot(
        np.arange(len(num_val_labels)),
        np.cumsum(num_val_labels),
        "b-",
        linewidth=1,
        label="val cumulative dist.",
    )
    ax0.plot(
        np.arange(len(num_val_labels)),
        np.cumsum(num_train_labels),
        "k-",
        linewidth=1,
        label="train cumulative dist.",
    )
    ax0.legend(loc="upper right")
    ax[0].legend()

    ax[1].bar(
        np.arange(len(num_train_labels)),
        np.log(num_train_labels),
        label="train labels",
    )
    ax[1].bar(
        np.arange(len(num_val_labels)),
        np.log(num_val_labels),
        label="val labels",
    )
    ax[1].set_title("number of training and validation labels per family (log scale)")
    ax[1].legend()

    # plot the cumulative number of labels
    ax1 = ax[1].twinx()
    ax1.plot(
        np.arange(len(num_train_labels)),
        np.cumsum(np.log(num_val_labels)),
        "b-",
        linewidth=1,
        label="val cumulative dist.",
    )
    ax1.plot(
        np.arange(len(num_train_labels)),
        np.cumsum(np.log(num_train_labels)),
        "k-",
        linewidth=1,
        label="train cumulative dist.",
    )
    ax1.legend(loc="upper right")

    ax[0].legend()
    ax[2].bar(
        np.arange(len(num_val_labels)),
        num_val_labels / num_train_labels,
        label="val labels",
    )
    ax[2].set_title(
        "ratio number of validation labels per family to number of train labels per family"
    )

    plt.savefig(handle_figure_path(figure_path))
    plt.close()


def number_of_sequences_per_unique_label_combination(hparams: dict, figure_path: str) -> None:

    train_labels_and_seq = load_sequences_and_labels(hparams["train_files"])
    val_labels_and_seq = load_sequences_and_labels(hparams["val_files"])

    train_labelset_to_count = defaultdict(int)
    for labelset, seq in train_labels_and_seq:
        if len(labelset) > 1:
            train_labelset_to_count["".join(sorted(labelset[1:]))] += 1

    val_labelset_to_count = defaultdict(int)
    for labelset, seq in val_labels_and_seq:
        if len(labelset) > 1:
            val_labelset_to_count["".join(sorted(labelset[1:]))] += 1

    val_counts = []
    train_counts = []

    for unique_label_combo in val_labelset_to_count.keys():
        val_counts.append(val_labelset_to_count[unique_label_combo])
        if unique_label_combo in train_labelset_to_count:
            train_counts.append(train_labelset_to_count[unique_label_combo])
        else:
            train_counts.append(0)

    val_counts = np.asarray(val_counts)
    train_counts = np.asarray(train_counts)
    idx = np.argsort(val_counts)
    val_counts = val_counts[idx]
    train_counts = train_counts[idx]

    fig, ax = plt.subplots(figsize=(13, 10), nrows=2)
    ax[0].bar(np.arange(len(train_counts)), train_counts, label="train")
    ax[0].bar(np.arange(len(val_counts)), val_counts, label="val")
    ax[0].set_xlabel("unique labelset")
    ax[0].set_ylabel("count")
    ax[0].set_ylim([0, 450])
    ax[0].set_title("intersection of validation and train labelsets")
    ax0 = ax[0].twinx()
    ax0.plot(
        np.arange(len(val_counts)),
        np.cumsum(val_counts),
        "b-",
        label="cumulative val dist",
    )
    ax0.plot(
        np.arange(len(train_counts)),
        np.cumsum(train_counts),
        "k-",
        label="cumulative train dist",
    )
    ax0.legend()

    ax[1].bar(np.arange(len(train_counts)), np.log(train_counts + 1), label="train")
    ax[1].bar(np.arange(len(val_counts)), np.log(val_counts + 1), label="val")
    ax[1].legend()

    ax[1].set_title(
        f"unique labelset (train, val) logged. pct of validation labelsets that don't have a match in train: {len(np.where(train_counts == 0)[0])/len(val_counts)}"
    )
    ax[1].set_xlabel("unique labelset")
    ax[1].set_ylabel("logged count")

    plt.savefig(handle_figure_path(figure_path))
    plt.close()


def parser():
    ap = ArgumentParser()
    sp = ap.add_subparsers(title="action", dest="command")
    spf = sp.add_parser(
        name="histogram_sequences",
        description="Histogram the number of sequences per family in the ingested fasta_files.",
    )
    spf.add_argument("fasta_files", nargs="+", help="fasta files(s) to histogram.")
    spf.add_argument("save_fig", type=str, help="where to save the figure")

    neighbor = sp.add_parser(
        name="neighborhood_labels",
        description="produce a figure describing "
        "the distribution of sequences with and without "
        "neighborhood labels. Accepts a single .yaml file"
        " or a set of fasta files.",
    )
    neighbor.add_argument(
        "files",
        nargs="+",
        help="list of files or a .yaml file containing train/val_file keys",
    )
    neighbor.add_argument("save_fig", type=str, help="where to save the figure")
    neighbor.add_argument(
        "--key",
        default="val_files",
        type=str,
        help="key to access in the .yaml file",
    )

    comp = sp.add_parser(
        name="compare_valid_train_labels",
        description="compare the distributions of training and validation labels."
        " Accepts a single .yaml file with train_files and val_files as keys",
    )
    comp.add_argument(
        "yaml_file",
        type=str,
        help=".yaml file containing train/val_file keys",
    )
    comp.add_argument("save_fig", type=str, help="where to save the figure")

    uniq = sp.add_parser(
        name="unique_label_combinations",
        description="plot the number of sequences per unique label combination (for neighborhood labels).",
    )

    uniq.add_argument(
        "yaml_file",
        type=str,
        help=".yaml file containing train/val_file keys",
    )
    uniq.add_argument("save_fig", type=str, help="where to save the figure")

    return ap


if __name__ == "__main__":

    p = parser()
    args = p.parse_args()
    if args.command == "histogram_sequences":
        histogram_number_of_seqs_per_family(args.fasta_files, args.save_fig)
    elif args.command == "neighborhood_labels":
        if len(args.files) == 1:
            file = args.files[0]
            if os.path.splitext(file)[1] == ".yaml":
                with open(file, "r") as src:
                    hparams = yaml.safe_load(src)
                files = hparams[args.key]
            elif os.path.splitext(file)[1] == ".fa":
                files = args.files
            else:
                raise ValueError(f"only accepts <.yaml, .fa>, got {args.files[0]}")
        else:
            files = args.files
        neighborhood_labels(files, args.save_fig)
    elif args.command == "compare_valid_train_labels":
        with open(args.yaml_file, "r") as src:
            hparams = yaml.safe_load(src)
        compare_valid_and_train_labels(hparams, args.save_fig)
    elif args.command == "unique_label_combinations":
        with open(args.yaml_file, "r") as src:
            hparams = yaml.safe_load(src)
        number_of_sequences_per_unique_label_combination(hparams, args.save_fig)

    else:
        p.print_help()
