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


def recall_for_each_significant_label(
    model,
    dataloader,
    name_to_class_code,
    figure_path,
    device="cuda",
    classification_threshold=0.5,
):
    """
    Which labels do the models tend to get right?
    Going to create a new dataloader that returns the index and order of labels.
    Then, aggregate stats for the 1st label; the second label; the third; and so on.
    :return:
    :rtype:
    """

    correct = defaultdict(int)
    total = defaultdict(int)

    j = 0
    with torch.no_grad():

        for features, masks, labels, string_labels in dataloader:

            features = features.to(device)
            masks = masks.to(device)
            pred = model.class_act(model(features, masks))

            pred[pred >= classification_threshold] = 1
            pred[pred < classification_threshold] = 0

            stdout.write(f"{j / len(dataloader)}\r")
            j += 1

            for seq, labelset in zip(pred, string_labels):
                for i, label in enumerate(labelset):
                    idx = name_to_class_code[label]
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
    ax1.legend()
    ax[0].legend()
    ax[0].set_xlabel("label rank (lower=lower e-value)")
    ax[0].set_ylabel("num labels")

    ax[1].bar(np.arange(len(correct)), correct / total, label="ratio of correct/total")
    ax[1].legend()

    plt.savefig(utils.handle_figure_path(figure_path))
    plt.close()


def primary_and_neighborhood_recall(
    model, dataloader, name_to_class_code, figure_path, device="cuda"
):
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

    with torch.no_grad():

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
                    fps = torch.sum(
                        seq[(labelvec != 1).bool() & (seq == 1).bool()]
                    ).item()
                    threshold_to_fps[threshold] += fps
                    for i, label in enumerate(labelset):
                        idx = name_to_class_code[label]
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
    ax1.set_ylabel(200)
    ax1.legend(loc="upper left")
    ax.legend()

    plt.savefig(utils.handle_figure_path(figure_path))
    plt.close()


# now I need to look at the recall of the model on a per-family basis;
# how does the validation performance change based on the number of sequences in the family at train time?
def _aggregate_family_wise_metrics(
    model,
    dataloader,
    name_to_class_code,
    classification_threshold,
    just_primary,
    just_neighborhood,
    device="cuda",
):
    j = 0
    family_to_tps = defaultdict(int)
    family_to_num_seq = defaultdict(int)

    with torch.no_grad():
        for features, masks, labels, string_labels in dataloader:

            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            pred = model.class_act(model(features, masks))
            stdout.write(f"{j / len(dataloader)}\r")
            j += 1
            for seq, labelset, labelvec in zip(pred, string_labels, labels):
                seq[seq >= classification_threshold] = 1
                if just_primary:
                    label = labelset[0]
                    idx = name_to_class_code[label]
                    family_to_num_seq[label] += 1
                    if seq[idx] == 1:
                        family_to_tps[label] += 1
                elif just_neighborhood:
                    labelset = labelset[1:]
                    for i, label in enumerate(labelset):
                        idx = name_to_class_code[label]
                        family_to_num_seq[label] += 1
                        if seq[idx] == 1:
                            family_to_tps[label] += 1
                else:
                    for i, label in enumerate(labelset):
                        idx = name_to_class_code[label]
                        family_to_num_seq[label] += 1
                        if seq[idx] == 1:
                            family_to_tps[label] += 1

    return family_to_tps, family_to_num_seq


def recall_per_family(
    model,
    train_files,
    val_dataloader,
    name_to_class_code,
    figure_path,
    classification_threshold=0.5,
    device="cuda",
    just_primary=False,
    just_neighborhood=False,
):

    train_labels_and_sequences = utils.load_sequences_and_labels(train_files)
    train_family_to_num_seq = defaultdict(int)

    for labelset, sequence in train_labels_and_sequences:
        if just_primary:
            train_family_to_num_seq[labelset[0]] += 1
        elif just_neighborhood:
            labelset = labelset[1:]
            for label in labelset:
                train_family_to_num_seq[label] += 1
        else:
            for label in labelset:
                train_family_to_num_seq[label] += 1

    val_family_to_tps, val_family_to_num_seq = _aggregate_family_wise_metrics(
        model,
        val_dataloader,
        name_to_class_code,
        classification_threshold,
        just_primary=just_primary,
        just_neighborhood=just_neighborhood,
        device=device,
    )
    train_num_seq = []
    val_num_seq = []
    val_tps = []

    for family in val_family_to_tps.keys():
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
    ax[0].legend()
    ax[1].legend()
    ax[2].plot(train_num_seq, val_tps / val_num_seq, "ko", markersize=2)
    ax[2].set_xlabel("number of sequences from family in train")
    ax[2].set_ylabel("% of validation seq. recovered")

    plt.savefig(utils.handle_figure_path(figure_path))
    plt.close()


def tps_above_threshold_torch(prediction_array, label_array, threshold):
    prediction_array[prediction_array >= threshold] = 1
    return torch.sum((prediction_array == 1) & (label_array == 1))


def fps_above_threshold_torch(prediction_array, label_array, threshold):
    prediction_array[prediction_array >= threshold] = 1
    return torch.sum((prediction_array == 1) & (label_array != 1))


def tps_above_threshold(prediction_array, label_array, threshold):
    prediction_array[prediction_array >= threshold] = 1
    return np.sum((prediction_array == 1) & (label_array == 1))


def fps_above_threshold(prediction_array, label_array, threshold):
    prediction_array[prediction_array >= threshold] = 1
    return np.sum((prediction_array == 1) & (label_array != 1))


def predict_and_plot_dataloader(
    model: pytorch_lightning.LightningModule,
    loader: torch.utils.data.DataLoader,
    figure_path: str,
    device: str = "cuda",
) -> None:
    i = 0
    tot = len(loader)
    thresholds = np.linspace(0, 1, 20)[::-1]
    threshold_to_tps_and_fps = {}
    for threshold in thresholds:
        threshold_to_tps_and_fps[threshold] = [0, 0]

    number_of_true_labels = 0
    number_sequences = 0
    number_neighborhood_labels = 0

    with torch.no_grad():
        for features, masks, label in loader:
            features = features.to(device)
            masks = masks.to(device)
            label = label.to(device)
            n = torch.sum(label == 1).item()
            number_of_true_labels += n
            number_neighborhood_labels += n - label.shape[0]
            number_sequences += features.shape[0]
            pred = model.class_act(model(features, masks))

            for threshold in thresholds:
                tps = tps_above_threshold_torch(
                    prediction_array=pred, label_array=label, threshold=threshold
                )
                fps = fps_above_threshold_torch(
                    prediction_array=pred, label_array=label, threshold=threshold
                )
                threshold_to_tps_and_fps[threshold][0] += tps.item()
                threshold_to_tps_and_fps[threshold][1] += fps.item()

            stdout.write(f"{i / tot}\r")
            i += 1

    thresholds = list(threshold_to_tps_and_fps.keys())
    tps = np.array([l[0] for l in list(threshold_to_tps_and_fps.values())])
    fps = np.array([l[1] for l in list(threshold_to_tps_and_fps.values())])

    print(number_sequences, number_of_true_labels, number_neighborhood_labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(thresholds, tps / number_of_true_labels, "bo", label="tps recovered")
    ax.plot(
        thresholds,
        tps / number_sequences,
        "co",
        label="number of family labels recovered",
    )
    ax.set_xlabel("sigmoid threshold")
    ax.set_ylabel("percent of tps recovered")
    ax.legend()
    ax1 = ax.twinx()
    ax1.plot(thresholds, fps / number_sequences, "ko", label="fps per seq.")
    ax1.set_ylabel("fps per sequence")
    plt.savefig(utils.handle_figure_path(figure_path))
    plt.close()


def create_parser():
    ap = ArgumentParser()
    sp = ap.add_subparsers(title="action", dest="command")
    recall_parser = sp.add_parser(
        name="recall",
        description="plot the recall and false positives passed at multiple sigmoid "
        "thresholds.",
    )
    recall_parser.add_argument("model_path")
    recall_parser.add_argument("hparams_path")
    recall_parser.add_argument("figure_path")
    recall_parser.add_argument("--key", default="val_files", type=str)

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

    primary_parser = sp.add_parser(
        name="primary_recall", description="plot the recall for each rank of label."
    )
    primary_parser.add_argument("model_path")
    primary_parser.add_argument("hparams_path")
    primary_parser.add_argument("figure_path")
    primary_parser.add_argument("--key", default="val_files", type=str)
    primary_parser.add_argument(
        "-c", "--classification_threshold", default=0.5, type=float
    )

    return ap


if __name__ == "__main__":

    argparser = create_parser()
    args = argparser.parse_args()

    model_path = args.model_path
    hparams_path = args.hparams_path

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=torch.device(dev))
    state_dict = checkpoint["state_dict"]
    state_dict["loss_func.pos_weight"] = torch.tensor(10)
    hparams["training"] = False

    model = models.Prot2Vec(**hparams).to(dev)
    success = model.load_state_dict(state_dict)
    model.eval()

    if args.command == "recall":
        files = hparams[args.key]
        name_to_class_code = hparams["name_to_class_code"]

        dataset = utils.SimpleSequenceIterator(files, name_to_class_code)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=utils.pad_batch
        )

        predict_and_plot_dataloader(model, dataloader, args.figure_path, device=dev)
    elif args.command == "recall_per_family":
        files = hparams["val_files"]
        name_to_class_code = hparams["name_to_class_code"]
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
            just_primary=args.just_primary,
            just_neighborhood=args.just_neighborhood,
            device=dev,
        )

    elif args.command == "primary_recall":
        files = hparams[args.key]
        name_to_class_code = hparams["name_to_class_code"]
        dataset = utils.RankingIterator(files, name_to_class_code)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=utils.pad_batch_with_labels
        )
        primary_and_neighborhood_recall(
            model,
            dataloader,
            name_to_class_code,
            args.figure_path,
            device=dev,
        )
    else:
        argparser.print_help()
