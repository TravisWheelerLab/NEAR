import os
import time
import pdb

import pytorch_lightning
import torch
import numpy as np
import yaml
import torchmetrics
import matplotlib.pyplot as plt
import faiss
import faiss.contrib.torch_utils
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict
from typing import Tuple
from sys import stdout
from random import shuffle
from pathlib import Path

import prefilter.utils as utils
import prefilter.models as models
from prefilter import AccessionIDToPfamName


@torch.no_grad()
def embed_logos(
    model, logo_path, accession_ids, device, embed_dim, add_all_logos=False
):
    mapping = AccessionIDToPfamName()
    logo_files = list(logo_path.glob("*.logo"))
    shuffle(logo_files)
    logos = np.zeros((len(accession_ids), embed_dim))
    in_already = set()
    i = 0

    # wait. since there isn't an hmm for those families, they will never be in accession ids
    # so it's fine. But annoying.
    for accession_id in sorted(accession_ids):
        name = mapping[accession_id]
        logo_file = logo_path / f"{name}.0.5-train.hmm.logo"
        if not logo_file.exists():
            raise ValueError(f"couldn't find {logo_file}")
        in_already.add(logo_file)
        embed = model(
            torch.as_tensor(utils.logo_from_file(logo_file))
            .to(device)
            .unsqueeze(0)
            .float()
        )
        logos[i] = embed.detach().cpu().numpy()
        i += 1

    if add_all_logos:
        print("adding in all logo files.")
        for f in logo_files:
            if f in in_already:
                logo_files.remove(f)

        decoy_logos = np.zeros((len(logo_files), embed_dim))
        i = 0
        for f in logo_files:
            print(f"adding in logo file {f}")
            if f not in in_already:
                embed = model(
                    torch.as_tensor(utils.logo_from_file(f))
                    .to(device)
                    .unsqueeze(0)
                    .float()
                )
                decoy_logos[i] = embed.detach().cpu().numpy()
                i += 1
            else:
                in_already.add(f)

        logos = np.concatenate((logos, decoy_logos), axis=0)

    return logos


def create_logo_index(logos, embed_dim, device="cpu"):
    index = faiss.IndexFlatIP(embed_dim)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(logos)
    return index


if __name__ == "__main__":

    logo_path = Path("/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/")
    hparams_path = (
        "models/contrastive/exps_mar31/with_emission/default/version_0/hparams.yaml"
    )
    model_path = "models/contrastive/exps_mar31/with_emission/default/version_0/checkpoints/epoch_2_3.546224.ckpt"
    figure_path = "1000_files_with_emission_sequences.png"
    embed_dim = 128
    batch_size = 32
    n_top = 1500
    add_all_logos = False
    analyze_train = True

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    checkpoint = torch.load(model_path, map_location=torch.device(dev))
    state_dict = checkpoint["state_dict"]
    hparams["training"] = False
    hparams["oversample_neighborhood_labels"] = False
    hparams["valid_files"] = []

    model = models.ResNet1d(**hparams).to(dev)

    success = model.load_state_dict(state_dict)
    print(f"{success} for model {model_path}")

    model.eval()

    valid_files = glob(
        "/home/tc229954/data/prefilter/pfam/seed/training_data/1000_file_subset/*valid.fa"
    )
    print(len(valid_files))
    valid_files = list(filter(lambda x: os.path.isfile(x), valid_files))

    name_to_class_code = utils.create_class_code_mapping(
        valid_files + hparams["fasta_files"]
    )

    accession_ids = list(name_to_class_code.keys())

    logo_embed = embed_logos(
        model,
        logo_path,
        accession_ids=accession_ids,
        device=dev,
        embed_dim=embed_dim,
        add_all_logos=add_all_logos,
    ).astype("float32")

    logo_idx = create_logo_index(logo_embed, embed_dim, device=dev)
    logo_embed = torch.as_tensor(logo_embed).to(dev)

    threshold_to_recall = defaultdict(int)
    threshold_to_neighborhood_recall = defaultdict(int)
    threshold_to_primary_recall = defaultdict(int)

    total_labelcount = 0
    primary_labelcount = 0
    neighborhood_labelcount = 0

    start = time.time()
    valid_files = [
        "/home/tc229954/data/prefilter/pfam/seed/training_data/benchmark/1000/valid_1000.fa"
    ]

    if analyze_train:
        dataset = utils.SequenceIterator(
            fasta_files=hparams["fasta_files"],
            name_to_class_code=name_to_class_code,
            max_labels_per_seq=100,
            evalue_threshold=1e-5,
        )
    else:
        dataset = utils.SequenceIterator(
            fasta_files=valid_files,
            name_to_class_code=name_to_class_code,
            max_labels_per_seq=100,
            evalue_threshold=1e-5,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=utils.pad_features_in_batch
    )
    for kk, (features, features_mask, labels) in enumerate(dataloader):
        print(f"{kk / len(dataloader):.3f}")
        predicted_embedding = model(torch.as_tensor(features).to(dev).float())
        # distance, indices
        distances, predicted_accession_ids = logo_idx.search(
            predicted_embedding,
            k=n_top if len(logo_embed) > n_top else logo_embed.shape[0],
        )
        # dot_prods = torch.matmul(logo_embed, predicted_embedding.T)
        # look @ overlap of sets?
        for i, (labelset, nearest_neighbor_set) in enumerate(
            zip(labels, predicted_accession_ids)
        ):
            for k, label in enumerate(labelset):
                total_labelcount += 1
                if k == 0:
                    primary_labelcount += 1
                else:
                    neighborhood_labelcount += 1

                acc_id = label[0]
                correct_idx = accession_ids.index(acc_id)
                topn = set()

                for j in range(0, nearest_neighbor_set.shape[-1]):
                    # without converting to numpy the set will
                    # contain torch.tensors and never match anything
                    topn.add(nearest_neighbor_set[j].item())
                    inn = correct_idx in topn
                    if k == 0:
                        threshold_to_primary_recall[j] += inn
                    else:
                        threshold_to_neighborhood_recall[j] += inn

                    threshold_to_recall[j] += inn

    topn = np.array(list(threshold_to_recall.keys()))
    recall = np.array(list(threshold_to_recall.values())) / total_labelcount

    primary_topn = np.array(list(threshold_to_primary_recall.keys()))
    primary_recall = (
        np.array(list(threshold_to_primary_recall.values())) / primary_labelcount
    )

    neighborhood_topn = np.array(list(threshold_to_neighborhood_recall.keys()))
    neighborhood_recall = (
        np.array(list(threshold_to_neighborhood_recall.values()))
        / neighborhood_labelcount
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot((topn / logo_embed.shape[0]) * 100, recall, "ro", label="total recall")
    ax.plot(
        (neighborhood_topn / logo_embed.shape[0]) * 100,
        neighborhood_recall,
        "bo",
        label="neighborhood recall",
    )
    ax.plot(
        (primary_topn / logo_embed.shape[0]) * 100,
        primary_recall,
        "ko",
        label="primary recall",
    )
    ax.set_xlabel("1 - percent filtered")
    ax.set_ylabel("% of labels recovererd")
    ax.invert_xaxis()
    ax.set_title(
        f"top1: {recall[1]:.3f}, {primary_recall[1]:.3f}, {neighborhood_recall[1]:.3f}\n"
        f"top5: {recall[5]:.3f}, {primary_recall[5]:.3f}, {neighborhood_recall[5]:.3f}\n"
        f"tp100: {recall[98]:.3f}, {primary_recall[98]:.3f}, {neighborhood_recall[98]:.3f}\n"
        f"total, primary, neighborhood"
    )

    plt.legend()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
