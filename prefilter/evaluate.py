# pylint: disable=no-member
import os
import pdb

import pytorch_lightning
import torch
import numpy as np
import yaml
import torchmetrics
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict
from typing import Tuple
from sys import stdout
from random import shuffle

import prefilter.utils as utils
import prefilter.models as models


@torch.no_grad()
def recall_for_each_significant_label(
    model,
    dataloader,
    name_to_class_code,
    figure_path,
    multi_prediction,
    device="cuda",
    classification_threshold=0.5,
):
    """
    Which rank of label do the models tend to get right?
    Produce a plot that summarizes on which labels the models do poorly. Primary labels are expected to have the best
    recall.
    :return: None
    :rtype: None
    """

    correct = defaultdict(int)
    total = defaultdict(int)

    j = 0
    for features, masks, labels, string_labels in dataloader:

        features = features.to(device)
        masks = masks.to(device)
        pred = model.class_act(model(features, masks))

        pred[pred >= classification_threshold] = 1

        stdout.write(f"{j / len(dataloader)}\r")
        j += 1

        for seq, labelset in zip(pred, string_labels):
            if multi_prediction:
                set_of_preds = set([t.item() for t in torch.where(seq == 1)[0]])
            for i, label in enumerate(labelset):
                if isinstance(label, list):
                    label = label[0]
                idx = name_to_class_code[label]
                if multi_prediction:
                    if idx in set_of_preds:
                        correct[i + 1] += 1
                else:
                    if seq[idx] == 1:
                        correct[i + 1] += 1
                total[i + 1] += 1

    fig, ax = plt.subplots(nrows=2, figsize=(13, 10))

    t = []
    c = []
    for num in total.keys():
        t.append(total[num])
        if num in correct:
            c.append(correct[num])
        else:
            c.append(0)

    correct = np.asarray(c)
    total = np.asarray(t)

    ax[0].bar(np.arange(len(total)), total, label="total")
    ax[0].bar(np.arange(len(correct)), correct, label="recalled")
    ax1 = ax[0].twinx()
    ax1.plot(np.arange(len(total)), np.cumsum(total), "k-", label="total")
    ax1.plot(np.arange(len(correct)), np.cumsum(correct), "b-", label="recalled")
    ax1.legend(loc="upper right")
    ax[0].legend(loc="upper left")
    ax[0].set_xlabel("label rank (lower=lower e-value)")
    ax[0].set_ylabel("num labels")

    ax[1].plot(
        np.arange(len(correct)), correct / total, "ko", label="ratio of correct/total"
    )
    ax[1].set_xlabel("label rank (lower=lower e-value)")
    ax[1].set_ylabel("% of label at rank K recalled")
    plt.suptitle("recall broken down by rank of label.")
    ax[1].legend(loc="upper left")

    if multi_prediction:
        figure_path, ext = os.path.splitext(figure_path)
        figure_path += "_multilabel" + ext

    plt.savefig(utils.handle_figure_path(figure_path))
    plt.close()


@torch.no_grad()
def primary_and_neighborhood_recall(
    model,
    dataloader,
    name_to_class_code,
    figure_path,
    multi_prediction,
    titlestr,
    device="cuda",
):
    """
    Plot the recall of the model for primary and neighborhood labels.
    :param model: model to evaluate
    :type model:
    :param dataloader: dataloader containing batched sequences/labels to evaluate
    :type dataloader:
    :param name_to_class_code: dictionary mapping pfam accession id to class code
    :type name_to_class_code:
    :param figure_path: where to save the figure
    :type figure_path:
    :param device: cuda or cpu - use the gpu or not?
    :type device:
    :return:
    :rtype:
    """
    thresholds = np.linspace(0.01, 1, 20)[::-1]
    threshold_to_tps_and_fps = {}

    for threshold in thresholds:
        threshold_to_tps_and_fps[threshold] = [0, 0]

    j = 0
    primary_recall = defaultdict(int)
    neighborhood_recall = defaultdict(int)
    threshold_to_fps = defaultdict(int)
    total = defaultdict(int)

    for threshold in thresholds:
        primary_recall[threshold] = 0
        neighborhood_recall[threshold] = 0
        threshold_to_fps[threshold] = 0

    for features, masks, labels, string_labels in dataloader:
        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        pred = model.class_act(model(features, masks))
        stdout.write(f"{j / len(dataloader)}\r")
        j += 1
        for seq, labelset, labelvec in zip(pred, string_labels, labels):
            for i, label in enumerate(labelset):
                total[i] += 1
            for threshold in thresholds:
                seq[seq >= threshold] = 1
                if multi_prediction:
                    # collapse set of predictions (n_classesxn_preds)
                    # down to the unique predictions
                    set_of_preds = set([t.item() for t in torch.where(seq == 1)[0]])
                    set_of_tps = set([t.item() for t in torch.where(labelvec == 1)[0]])
                    fps = len(set_of_preds - set_of_tps)
                else:
                    fps = torch.sum(
                        seq[(labelvec != 1).bool() & (seq == 1).bool()]
                    ).item()

                threshold_to_fps[threshold] += fps

                for i, label in enumerate(labelset):
                    if isinstance(label, list):
                        label = label[0]
                    idx = name_to_class_code[label]
                    if multi_prediction:
                        if idx in set_of_preds and i == 0:
                            primary_recall[threshold] += 1
                        elif idx in set_of_preds and i != 0:
                            neighborhood_recall[threshold] += 1
                    else:
                        if seq[idx] == 1 and i == 0:
                            primary_recall[threshold] += 1
                        elif seq[idx] == 1 and i != 0:
                            neighborhood_recall[threshold] += 1

    total_primary_labels = total[0]
    total_neighborhood_labels = sum([total[i] for i in range(1, len(total))])
    primary_recalled_labels = np.asarray(list(primary_recall.values()))
    neigh_recalled_labels = np.asarray(list(neighborhood_recall.values()))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(
        thresholds,
        primary_recalled_labels / total_primary_labels,
        "ko",
        label="primary recall",
    )
    ax.plot(
        thresholds,
        neigh_recalled_labels / total_neighborhood_labels,
        "bo",
        label="neighborhood recall",
    )
    ax.plot(
        thresholds,
        (primary_recalled_labels + neigh_recalled_labels)
        / (total_primary_labels + total_neighborhood_labels),
        "ro",
        label="total recall",
    )
    ax1 = ax.twinx()
    ax1.plot(
        thresholds,
        np.asarray(list(threshold_to_fps.values())) / total_primary_labels,
        "co",
        label="fps per seq",
    )
    ax1.set_ylabel("false positives per sequence")
    ax1.legend(loc="upper right")
    ax.legend(loc="upper left")
    ax.set_xlabel("sigmoid threshold")
    ax.set_ylim([0, 1])
    ax.set_ylabel("% of true positives recalled")
    ax.set_title("primary and neighborhood recall as a function of sigmoid threshold")

    if titlestr is not None:
        ax.set_title(f"_{titlestr}")

    if multi_prediction:
        figure_path, ext = os.path.splitext(figure_path)
        figure_path += "_multilabel" + ext

    plt.savefig(utils.handle_figure_path(figure_path))

    plt.close()


# now I need to look at the recall of the model on a per-family basis;
# how does the validation performance change based on the number of sequences in the family at train time?
@torch.no_grad()
def _aggregate_family_wise_metrics(
    model,
    dataloader,
    name_to_class_code,
    classification_threshold,
    just_primary,
    just_neighborhood,
    multi_prediction,
    device="cuda",
):
    j = 0
    family_to_tps = defaultdict(int)
    family_to_num_seq = defaultdict(int)

    for features, masks, labels, string_labels in dataloader:

        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        pred = model.class_act(model(features, masks))
        stdout.write(f"{j / len(dataloader)}\r")
        j += 1
        for seq, labelset, labelvec in zip(pred, string_labels, labels):
            seq[seq >= classification_threshold] = 1

            if multi_prediction:
                set_of_preds = set([t.item() for t in torch.where(seq == 1)[0]])

            if just_primary:
                label = labelset[0]
                if isinstance(label, list):
                    label = label[0]
                idx = name_to_class_code[label]
                family_to_num_seq[label] += 1
                if multi_prediction:
                    if idx in set_of_preds:
                        family_to_tps[label] += 1
                else:
                    if seq[idx] == 1:
                        family_to_tps[label] += 1

            elif just_neighborhood:
                labelset = labelset[1:]
                for i, label in enumerate(labelset):
                    if isinstance(label, list):
                        label = label[0]
                    idx = name_to_class_code[label]
                    family_to_num_seq[label] += 1

                    if multi_prediction:
                        if idx in set_of_preds:
                            family_to_tps[label] += 1
                    else:
                        if seq[idx] == 1:
                            family_to_tps[label] += 1

            else:
                for i, label in enumerate(labelset):
                    if isinstance(label, list):
                        label = label[0]
                    idx = name_to_class_code[label]
                    family_to_num_seq[label] += 1
                    if multi_prediction:
                        if idx in set_of_preds:
                            family_to_tps[label] += 1
                    else:
                        if seq[idx] == 1:
                            family_to_tps[label] += 1

    return family_to_tps, family_to_num_seq


def recall_per_family(
    model,
    train_files,
    val_dataloader,
    name_to_class_code,
    figure_path,
    emission_files=None,
    classification_threshold=0.5,
    device="cuda",
    just_primary=False,
    just_neighborhood=False,
    multi_prediction=False,
):
    """
    Plot the recall on a family-wise basis; try to answer "Does model performance increase when there are more sequences
    in train from a given family?"
    :param multi_prediction:
    :type multi_prediction:
    :param emission_files: files containing emission sequences
    :type emission_files: List[str]
    :param model: model to evaluate
    :type model:
    :param train_files: files containing train sequences
    :type train_files: List[str]
    :param val_dataloader: dataloader (batched) with val seq+labels (ingested into model)
    :type val_dataloader:
    :param name_to_class_code: dictionary mapping pfam accession id to class code
    :type name_to_class_code: Dict[str, int]
    :param figure_path: where to save the figure
    :type figure_path: str
    :param classification_threshold: sigmoid values above this will be shunted to 1 (a positive classification)
    :type classification_threshold: float
    :param device: cuda or cpu
    :type device: str
    :param just_primary: just analyze primary labels
    :type just_primary: bool
    :param just_neighborhood: just analyze neighborhood labels
    :type just_neighborhood: bool
    :return: None
    :rtype: None
    """

    train_labels_and_sequences = utils.load_sequences_and_labels(train_files)
    train_family_to_num_seq = defaultdict(int)

    for labelset, sequence in train_labels_and_sequences:
        if just_primary:
            if isinstance(labelset[0], list):
                train_family_to_num_seq[labelset[0][0]] += 1
            else:
                train_family_to_num_seq[labelset[0]] += 1
        elif just_neighborhood:
            labelset = labelset[1:]
            for label in labelset:
                if isinstance(label, list):
                    label = label[0]
                train_family_to_num_seq[label] += 1
        else:
            for label in labelset:
                if isinstance(label, list):
                    label = label[0]
                train_family_to_num_seq[label] += 1

    if emission_files is not None:
        # load the emission sequences and count the number of instances per family.
        # to plot the relationship b/t train seq and validation performance.
        emission_labels_and_sequences = utils.load_sequences_and_labels(emission_files)
        emission_family_to_num_seq = defaultdict(int)

        for labelset, sequence in emission_labels_and_sequences:
            for label in labelset:
                if isinstance(label, list):
                    label = label[0]
                emission_family_to_num_seq[label] += 1

    val_family_to_tps, val_family_to_num_seq = _aggregate_family_wise_metrics(
        model,
        val_dataloader,
        name_to_class_code,
        classification_threshold,
        just_primary=just_primary,
        just_neighborhood=just_neighborhood,
        multi_prediction=multi_prediction,
        device=device,
    )
    train_num_seq = []
    val_num_seq = []
    val_tps = []

    for family in val_family_to_tps.keys():
        if emission_files is not None:
            if family in emission_family_to_num_seq:
                emission_seq = emission_family_to_num_seq[family]
            else:
                emission_seq = 0
            train_num_seq.append(train_family_to_num_seq[family] + emission_seq)
        else:
            train_num_seq.append(train_family_to_num_seq[family])

        val_num_seq.append(val_family_to_num_seq[family])
        val_tps.append(val_family_to_tps[family])

    train_num_seq = np.asarray(train_num_seq)
    val_num_seq = np.asarray(val_num_seq)
    val_tps = np.asarray(val_tps)

    idx = np.argsort(train_num_seq)
    train_num_seq = train_num_seq[idx]
    val_num_seq = val_num_seq[idx]
    val_tps = val_tps[idx]

    fig, ax = plt.subplots(figsize=(13, 10), nrows=3)
    ax[0].set_title("number of sequences per family in train and validation, unlogged")
    ax[0].bar(np.arange(len(train_num_seq)), train_num_seq, label="seq. in train")
    ax[0].bar(np.arange(len(val_num_seq)), val_num_seq, label="seq. in val")
    ax[0].bar(np.arange(len(val_tps)), val_tps, label="val tps")

    ax[1].set_title("number of sequences per family in train and validation, logged")
    ax[1].bar(
        np.arange(len(train_num_seq)), np.log(train_num_seq), label="seq. in train"
    )
    ax[1].bar(np.arange(len(val_num_seq)), np.log(val_num_seq), label="seq. in val")
    ax[1].bar(np.arange(len(val_tps)), np.log(val_tps), label="val tps")
    ax[1].set_xlabel("family id (sorted by membership)")
    ax[1].set_ylabel("ln(count)")
    ax[0].set_ylabel("count")

    ax[0].legend()
    ax[1].legend()

    ax[2].plot(train_num_seq, val_tps / val_num_seq, "ko", markersize=2)
    ax[2].set_xlabel("number of sequences from family in train")
    ax[2].set_ylabel("% of validation sequences recovered from family")

    if just_primary:
        title = "just primary labels."
    elif just_neighborhood:
        title = "just neighborhood labels."
    else:
        title = "all labels."

    plt.suptitle(f"per-family performance, {title}")

    if multi_prediction:
        figure_path, ext = os.path.splitext(figure_path)
        figure_path += "_multilabel" + ext

    plt.savefig(utils.handle_figure_path(figure_path))

    plt.close()


def tps_above_threshold_torch(prediction_array, label_array, threshold):
    """
    Calculate true positives above threshold in a prediction tensor.
    :param prediction_array:
    :type prediction_array:
    :param label_array:
    :type label_array:
    :param threshold:
    :type threshold:
    :return:
    :rtype:
    """
    prediction_array[prediction_array >= threshold] = 1
    return torch.sum((prediction_array == 1) & (label_array == 1))


def fps_above_threshold_torch(prediction_array, label_array, threshold):
    """
    Calculate true positives above threshold in a prediction tensor.
    :param prediction_array:
    :type prediction_array:
    :param label_array:
    :type label_array:
    :param threshold:
    :type threshold:
    :return:
    :rtype:
    """
    prediction_array[prediction_array >= threshold] = 1
    return torch.sum((prediction_array == 1) & (label_array != 1))


def tps_above_threshold(prediction_array, label_array, threshold):
    prediction_array[prediction_array >= threshold] = 1
    return np.sum((prediction_array == 1) & (label_array == 1))


def fps_above_threshold(prediction_array, label_array, threshold):
    prediction_array[prediction_array >= threshold] = 1
    return np.sum((prediction_array == 1) & (label_array != 1))


def create_parser():
    ap = ArgumentParser()
    sp = ap.add_subparsers(title="action", dest="command")
    recall_parser = sp.add_parser(
        name="recall",
        description="plot the recall and false positives passed at multiple sigmoid "
        "thresholds. Break up into primary and neighborhood labels.",
    )
    recall_parser.add_argument("model_path")
    recall_parser.add_argument("hparams_path")
    recall_parser.add_argument("figure_path")
    recall_parser.add_argument("-e", "--emission_sequence_path", default=None)
    recall_parser.add_argument("-t", "--titlestr", default=None)
    recall_parser.add_argument("--key", default="val_files", type=str)
    recall_parser.add_argument("--multilabel", action="store_true")

    recall_per_fam_parser = sp.add_parser(
        name="recall_per_family",
        description="plot the comparison between train/val dists. and their "
        "performance as a function of number of train labels",
    )
    recall_per_fam_parser.add_argument("model_path")
    recall_per_fam_parser.add_argument("hparams_path")
    recall_per_fam_parser.add_argument("figure_path")
    recall_per_fam_parser.add_argument(
        "-c", "--classification_threshold", default=0.5, type=float
    )
    recall_per_fam_parser.add_argument("--just_primary", action="store_true")
    recall_per_fam_parser.add_argument("--just_neighborhood", action="store_true")
    recall_per_fam_parser.add_argument("--multilabel", action="store_true")

    primary_parser = sp.add_parser(
        name="ranked_recall", description="plot the recall for each rank of label."
    )
    primary_parser.add_argument("model_path")
    primary_parser.add_argument("hparams_path")
    primary_parser.add_argument("figure_path")
    primary_parser.add_argument("--multilabel", action="store_true")
    primary_parser.add_argument("--key", default="val_files", type=str)
    primary_parser.add_argument(
        "-c", "--classification_threshold", default=0.5, type=float
    )

    return ap


def main():
    argparser = create_parser()
    args = argparser.parse_args()

    model_path = args.model_path
    hparams_path = args.hparams_path

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    dev = "cuda:2" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=torch.device(dev))
    state_dict = checkpoint["state_dict"]
    hparams["training"] = False
    state_dict["loss_func.weight"] = torch.tensor(10)

    model = models.Prot2Vec(**hparams).to(dev)
    success = model.load_state_dict(state_dict)
    print(success)

    model.eval()

    if args.command == "recall":
        files = hparams[args.key]
        if args.emission_sequence_path is not None:
            emission_files = glob(os.path.join(args.emission_sequence_path, "*fa"))
            files = emission_files
        name_to_class_code = hparams["name_to_class_code"]
        dataset = utils.RankingIterator(files, name_to_class_code)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            collate_fn=utils.pad_batch_with_labels,
            shuffle=True,
        )
        primary_and_neighborhood_recall(
            model,
            dataloader,
            name_to_class_code,
            args.figure_path,
            titlestr=args.titlestr,
            multi_prediction=args.multilabel,
            device=dev,
        )
    elif args.command == "recall_per_family":
        files = hparams["val_files"]
        name_to_class_code = hparams["name_to_class_code"]
        # None or List[str].
        emission_files = hparams.get("emission_files")

        dataset = utils.RankingIterator(files, name_to_class_code)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=utils.pad_batch_with_labels
        )

        if args.just_neighborhood and args.just_primary:
            raise ValueError("Can't specify just primary and just neighborhood.")

        recall_per_family(
            model,
            hparams["train_files"],
            dataloader,
            name_to_class_code,
            args.figure_path,
            emission_files=emission_files,
            just_primary=args.just_primary,
            just_neighborhood=args.just_neighborhood,
            multi_prediction=args.multilabel,
            device=dev,
        )

    elif args.command == "ranked_recall":
        files = hparams[args.key]
        name_to_class_code = hparams["name_to_class_code"]
        dataset = utils.RankingIterator(files, name_to_class_code)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=utils.pad_batch_with_labels
        )
        recall_for_each_significant_label(
            model,
            dataloader,
            name_to_class_code,
            args.figure_path,
            args.multilabel,
            classification_threshold=args.classification_threshold,
            device=dev,
        )
    else:
        argparser.print_help()


if __name__ == "__main__":
    main()
