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

__all__ = ["embed_logos", "create_logo_index"]


@torch.no_grad()
def embed_logos(
    model,
    logo_path,
    accession_ids,
    device,
    embed_dim,
    add_all_logos=False,
    batch_size=32,
):
    mapping = AccessionIDToPfamName()
    logo_files = list(logo_path.glob("*.logo"))
    shuffle(logo_files)
    in_already = set()
    # wait. since there isn't an hmm for those families, they will never be in accession ids
    # so it's fine. But annoying.
    sorted_logo_files = []
    for accession_id in sorted(accession_ids):
        name = mapping[accession_id]
        logo_file = logo_path / f"{name}.0.5-train.hmm.logo"
        if not logo_file.exists():
            raise ValueError(f"couldn't find {logo_file}")
        in_already.add(logo_file)
        sorted_logo_files.append(logo_file)

    if add_all_logos:
        for f in logo_files:
            if f in in_already:
                logo_files.remove(f)

        for f in logo_files:
            if f not in in_already:
                sorted_logo_files.append(f)
            else:
                in_already.add(f)

    batcher = utils.LogoBatcher(sorted_logo_files, {})
    batcher = torch.utils.data.DataLoader(
        batcher,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.pad_features_in_batch,
    )

    logo_embeddings = torch.zeros((len(sorted_logo_files), embed_dim), device=device)
    i = 0
    with torch.no_grad():
        for features, features_mask, _ in batcher:
            predicted_embeddings = model(features.to(device), features_mask.to(device))
            logo_embeddings[i : i + batch_size] = predicted_embeddings
            i += batch_size

    print("Num logos:", len(logo_embeddings))

    return logo_embeddings.float()


def create_logo_index(logos, embed_dim, device="cpu"):
    index = faiss.IndexFlatIP(embed_dim)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        # 0 is the index of the GPU. Since we're always using slurm
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(logos)
    return index


def main(
    model,
    fasta_files,
    logo_path,
    logo_accession_ids,
    device,
    add_all_logos,
    embedding_dimension,
    benchmarking,
    n_top,
    figure_path,
):

    print("embedding logos.")

    logo_embed = embed_logos(
        model,
        logo_path,
        accession_ids=logo_accession_ids,
        device=device,
        embed_dim=embedding_dimension,
        add_all_logos=add_all_logos,
    )

    logo_idx = create_logo_index(logo_embed, embedding_dimension, device=device)
    logo_embed = torch.as_tensor(logo_embed).to(dev)
    print(logo_embed.shape)

    threshold_to_recall = defaultdict(int)
    threshold_to_neighborhood_recall = defaultdict(int)
    threshold_to_primary_recall = defaultdict(int)

    total_labelcount = 0
    primary_labelcount = 0
    neighborhood_labelcount = 0

    start = time.time()
    dataset = utils.SequenceIterator(
        fasta_files=fasta_files,
        name_to_class_code=name_to_class_code,
        max_labels_per_seq=100,
        evalue_threshold=1e-5,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=utils.pad_features_in_batch,
        num_workers=0,
    )
    for kk, (features, features_mask, labels) in enumerate(dataloader):
        print(f"{kk / len(dataloader):.3f}")
        predicted_embedding = model(torch.as_tensor(features).to(device).float())
        # distance, indices
        distances, predicted_accession_ids = logo_idx.search(
            predicted_embedding, k=n_top
        )

        if benchmarking:
            continue

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
    if benchmarking:
        end = time.time()
        print(f"time taken to evaluate: {end-start}s")
        exit()
    else:
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
        ax.plot(
            (1 - (topn / logo_embed.shape[0])) * 100, recall, "ro", label="total recall"
        )
        ax.plot(
            (1 - (neighborhood_topn / logo_embed.shape[0])) * 100,
            neighborhood_recall,
            "bo",
            label="neighborhood recall",
        )
        ax.plot(
            (1 - (primary_topn / logo_embed.shape[0])) * 100,
            primary_recall,
            "ko",
            label="primary recall",
        )
        ax.set_xlabel("percent filtered")
        ax.set_ylabel("% of labels recovererd")
        ax.set_title(
            f"top1: {recall[1]*100:.3f}, {primary_recall[1]*100:.3f}, {neighborhood_recall[1]*100:.3f}\n"
            f"top5: {recall[5]*100:.3f}, {primary_recall[5]*100:.3f}, {neighborhood_recall[5]*100:.3f}\n"
            f"tp100: {recall[100]*100:.3f}, {primary_recall[100]*100:.3f}, {neighborhood_recall[100]*100:.3f}\n"
            f"tp{n_top}: {recall[n_top-1]*100:.3f}, {primary_recall[n_top-1]*100:.3f}, {neighborhood_recall[n_top-1]*100:.3f}\n"
            f"total, primary, neighborhood"
        )

        plt.legend()
        plt.savefig(utils.handle_figure_path(figure_path), bbox_inches="tight")
        plt.close()


def create_parser():
    ap = ArgumentParser()
    ap.add_argument("model_root_dir")
    ap.add_argument("model_name")
    ap.add_argument("figure_path")
    ap.add_argument(
        "--logo_path",
        default=Path("/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/"),
    )
    ap.add_argument("-b", "--benchmark", action="store_true")
    ap.add_argument(
        "-a",
        "--add_all_families",
        action="store_true",
        help="evaluate using all logos in" "--logo_path",
    )
    ap.add_argument("-e", "--embed_dim", type=int, default=128)
    ap.add_argument("-bsz", "--batch_size", type=int, default=32)
    ap.add_argument(
        "--debug",
        action="store_true",
        help="reduce the number of logo files" "for quick debugging.",
    )
    ap.add_argument(
        "-n", "--n_top", type=int, default=1000, help="num nearest neighbors to include"
    )
    ap.add_argument(
        "-t",
        "--analyze_train",
        action="store_true",
        help="run the analysis on the train set",
    )
    ap.add_argument(
        "-i",
        "--include_emission",
        action="store_true",
        help="include emission files when evaluating. Only valid when --analyze_train is set.",
    )

    return ap


if __name__ == "__main__":

    args = create_parser().parse_args()

    logo_path = args.logo_path
    hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
    model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
    figure_path = args.figure_path
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    n_top = args.n_top
    add_all_logos = args.add_all_families
    benchmarking = args.benchmark

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(dev)

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    checkpoint = torch.load(model_path, map_location=torch.device(dev))
    state_dict = checkpoint["state_dict"]
    hparams["training"] = False

    stime = time.time()
    model = models.ResNet1d(**hparams).to(dev)

    success = model.load_state_dict(state_dict)
    etime = time.time()
    print(f"{success} for model {model_path}, took {etime-stime}")

    model.eval()

    if args.analyze_train:
        fasta_files = hparams["fasta_files"]
        if args.include_emission:
            fasta_files += hparams["emission_files"]
    else:
        fasta_files = hparams["valid_files"]

    name_to_class_code = utils.create_class_code_mapping(
        set(fasta_files + hparams["fasta_files"])
    )

    accession_ids = list(name_to_class_code.keys())
    if args.debug:
        accession_ids = accession_ids[:2]
    main(
        model=model,
        fasta_files=fasta_files,
        logo_path=logo_path,
        logo_accession_ids=accession_ids,
        device=dev,
        add_all_logos=add_all_logos,
        embedding_dimension=embed_dim,
        benchmarking=benchmarking,
        n_top=n_top,
        figure_path=figure_path,
    )
