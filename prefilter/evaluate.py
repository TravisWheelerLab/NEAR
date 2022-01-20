# pylint: disable=no-member
import os

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

    with torch.no_grad():
        for features, masks, label in loader:
            features = features.to(device)
            masks = masks.to(device)
            label = label.to(device)
            number_of_true_labels += torch.sum(label == 1).item()
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

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(thresholds, tps / number_of_true_labels, "bo", label="tps recovered")
    ax.set_xlabel("sigmoid threshold")
    ax.set_ylabel("percent of tps recovered")
    ax1 = ax.twinx()
    ax1.plot(thresholds, fps / number_sequences, "ko", label="fps per seq.")
    ax1.set_ylabel("fps per sequence")
    plt.savefig(utils.handle_figure_path(figure_path))
    plt.close()


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("hparams_path")
    ap.add_argument("figure_path")
    ap.add_argument("--key", default="val_files", type=str)
    args = ap.parse_args()

    model_path = args.model_path
    hparams_path = args.hparams_path

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=torch.device(dev))
    state_dict = checkpoint["state_dict"]
    state_dict["loss_func.pos_weight"] = torch.tensor(10)

    model = models.Prot2Vec(**hparams, training=False).to(dev)
    success = model.load_state_dict(state_dict)
    model.eval()

    files = hparams[args.key]
    name_to_class_code = hparams["name_to_class_code"]

    dataset = utils.SimpleSequenceIterator(files, name_to_class_code)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, collate_fn=utils.pad_batch
    )

    predict_and_plot_dataloader(model, dataloader, args.figure_path)
