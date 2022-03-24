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
from pathlib import Path

import prefilter.utils as utils
import prefilter.models as models
from prefilter import AccessionIDToPfamName


@torch.no_grad()
def embed_logos(model, logo_path, accession_ids, device, embed_dim):
    mapping = AccessionIDToPfamName()
    logos = np.zeros((len(accession_ids), embed_dim))
    for i, accession_id in enumerate(accession_ids):
        name = mapping[accession_id]
        logo_file = logo_path / f"{name}.0.5-train.hmm.logo"
        if not logo_file.exists():
            raise ValueError(f"couldn't find {logo_file}")
        embed = model(
            torch.as_tensor(utils.logo_from_file(logo_file))
            .to(device)
            .unsqueeze(0)
            .float()
        )
        logos[i] = embed.detach().cpu().numpy()
    return logos


if __name__ == "__main__":

    fasta_files = glob("/home/tc229954/max_hmmsearch/200_file_subset/*valid.fa")
    logo_path = Path("/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/")
    hparams_path = "models/contrastive/exps_mar23/default/version_0/hparams.yaml"
    model_path = "models/contrastive/exps_mar23/default/version_0/checkpoints/ckpt-4838-0.35327672958374023.ckpt"
    embed_dim = 128

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    checkpoint = torch.load(model_path, map_location=torch.device(dev))
    state_dict = checkpoint["state_dict"]
    hparams["training"] = False

    model = models.ResNet1d(**hparams).to(dev)
    success = model.load_state_dict(state_dict)
    print(f"{success} for model {model_path}")
    model.eval()

    valid_files = [f.replace("-train.fa", "-valid.fa") for f in hparams["fasta_files"]]
    valid_files = list(filter(lambda x: os.path.isfile(x), valid_files))

    name_to_class_code = utils.create_class_code_mapping(hparams["fasta_files"])
    accession_ids = list(name_to_class_code.keys())

    logo_embed = (
        torch.as_tensor(
            embed_logos(
                model,
                logo_path,
                accession_ids=accession_ids,
                device=dev,
                embed_dim=embed_dim,
            )
        )
        .to(dev)
        .float()
    )

    dataset = utils.SequenceIterator(
        hparams["fasta_files"],
        name_to_class_code=name_to_class_code,
        max_labels_per_seq=100,
        evalue_threshold=1e-5,
    )

    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=utils.pad_features_in_batch
    )
    threshold_to_recall = defaultdict(int)
    threshold_to_neighborhood_recall = defaultdict(int)
    threshold_to_primary_recall = defaultdict(int)
    shuffle(accession_ids)

    total_labelcount = 0
    primary_labelcount = 0
    neighborhood_labelcount = 0
    for features, features_mask, labels in dataloader:
        predicted_embedding = model(torch.as_tensor(features).to(dev).float())

        nearest_neighbors = torch.matmul(logo_embed, predicted_embedding.T).squeeze()
        print(nearest_neighbors)
        exit()

        nearest_neighbors_idx = torch.argsort(nearest_neighbors, dim=0).cpu().numpy()
        # iterate over indices backwards
        for i, labelset in enumerate(labels):
            for k, label in enumerate(labelset):
                total_labelcount += 1
                if k == 0:
                    primary_labelcount += 1
                else:
                    neighborhood_labelcount += 1

                acc_id = label[0]
                correct_idx = accession_ids.index(acc_id)
                # iterate backwards over the indices of nearest neighbors
                for j in range(1, nearest_neighbors_idx.shape[0]):
                    topn = set(nearest_neighbors_idx[:, i][-j:])
                    inn = correct_idx in topn
                    if k == 0:
                        threshold_to_primary_recall[j] += inn
                    else:
                        threshold_to_neighborhood_recall[j] += inn

                    threshold_to_recall[j] += inn
                break

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

    ax.plot(topn, recall, "ro", label="total recall")
    ax.plot(neighborhood_topn, neighborhood_recall, "bo", label="neighborhood recall")
    ax.plot(primary_topn, primary_recall, "ko", label="primary recall")
    ax.set_xlabel("number passed threshold")
    ax.set_ylabel("% of labels recovererd")
    ax.invert_xaxis()
    plt.legend()
    plt.savefig("random.png", bbox_inches="tight")
    plt.close()
