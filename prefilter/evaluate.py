import itertools
import os
import pdb
import re
import time
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from sys import stdout
from typing import List

import esm
import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch_lightning import seed_everything

import prefilter.models as model_utils
import prefilter.utils as utils
from prefilter.models import DotProdModel, ResidualBlock, ResNet


def non_default_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([torch.tensor(b[1]) for b in batch]),
        [b[2] for b in batch],
    )


def compute_accuracy(
    query_dataset,
    cluster_rep_index,
    cluster_rep_labels,
    trained_model,
    n_neighbors,
    pretrained_transformer,
    index_device,
    normalize,
    device="cuda",
):
    total_sequences = 0
    thresholds = [1, 2, 3, 5, 10, 20, 100, 200]
    topn = defaultdict(int)

    if pretrained_transformer:
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

    for j, (features, labels, sequences) in enumerate(query_dataset):
        if pretrained_transformer:
            embeddings = trained_model(
                features.to(device), repr_layers=[33], return_contacts=False
            )

            embed = []
            for k, seq in enumerate(sequences):
                embed.append(embeddings["representations"][33][k, 1 : len(seq) + 1])
            embeddings = embed
        else:
            if features.shape[0] == 128:
                features = features.unsqueeze(0)
            embeddings = trained_model(features.to(device)).transpose(-1, -2)

        stdout.write(f"{j / len(query_dataset):.3f}\r")
        # searching each sequence separately against the index is probably slow.
        for label, sequence in zip(labels, embeddings):
            if not isinstance(label, int):
                label = int(label)
            total_sequences += 1
            if normalize:
                sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()
            else:
                sequence = sequence.contiguous()

            predicted_labels, counts = utils.most_common_matches(
                cluster_rep_index,
                cluster_rep_labels,
                sequence,
                n_neighbors,
                index_device,
            )

            top_preds = predicted_labels[np.argsort(counts)]

            for n in thresholds:
                top_pred = top_preds[-n:]
                if label in set(top_pred):
                    topn[n] += 1

    correct_counts = np.asarray([topn[t] for t in thresholds])
    thresholds = ", ".join([str(t) for t in thresholds])
    percent_correct = ", ".join([f"{c / total_sequences:.3f}" for c in correct_counts])
    _, total_families = np.unique(cluster_rep_labels, return_counts=True)

    print(
        f"{thresholds}\n",
        f"{percent_correct}\n" f"Total families searched: {len(total_families)}",
        f"Total sequences: {total_sequences}",
    )


@torch.no_grad()
def main(fasta_files):
    parser = utils.create_parser()
    args = parser.parse_args()
    seq_len = args.seq_len
    batch_size = args.batch_size
    index_device = args.index_device

    embed_dim = args.embed_dim
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # get model files
    if args.pretrained_transformer:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        model.eval()  # disables dropout for deterministic results
        embed_dim = 1280
    elif args.msa_transformer:
        msa_transformer, _ = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        msa_transformer.eval()
        msa_transformer.requires_grad_ = False
        hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
        model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
        with open(hparams_path, "r") as src:
            hparams = yaml.safe_load(src)
        # Set up model and dataset
        if not os.path.isfile(model_path):
            raise ValueError(f"No model found at {model_path}")
        model, _ = utils.load_model(model_path, hparams, dev)

    else:
        hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
        model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
        with open(hparams_path, "r") as src:
            hparams = yaml.safe_load(src)
        # Set up model and dataset
        if not os.path.isfile(model_path):
            raise ValueError(f"No model found at {model_path}")
        model, _ = utils.load_model(model_path, hparams, dev)

    if args.msa_transformer:
        family_model = msa_transformer
    else:
        family_model = model

    if args.msa_transformer:
        iterator = utils.MSAClusterIterator(afa_files=fasta_files, seq_len=seq_len)
    elif args.profmark:
        iterator = utils.ProfmarkDataset(
            "test",
            profmark_dir="/home/tc229954/data/prefilter/pfam/seed/profmark",
            n_seq_per_target_family=args.n_seq_per_target_family,
            seq_len=seq_len,
        )
    else:
        iterator = utils.ClusterIterator(
            fasta_files,
            seq_len,
            representative_index=0,
            include_all_families=args.include_all_families,
            n_seq_per_target_family=args.n_seq_per_target_family,
            transformer=args.pretrained_transformer,
            return_alignments=args.plot_recall_and_pid,
        )

    rep_seqs, rep_labels = iterator.get_cluster_representatives()
    rep_gapped_seqs = iterator.seed_alignments
    # stack the seed sequences.
    rep_embeddings, rep_labels = utils.compute_cluster_representative_embeddings(
        rep_seqs,
        rep_labels,
        family_model,
        cnn_model=model,
        normalize=args.normalize_embeddings,
        device=dev,
        pretrained_transformer=args.pretrained_transformer,
        msa_transformer=args.msa_transformer,
    )
    print(
        f"{rep_embeddings.shape[0]} AA embeddings in target DB. Embedding dimension: {rep_embeddings.shape[1]}"
    )

    # create an index
    index = utils.create_faiss_index(
        rep_embeddings,
        embed_dim,
        device=index_device,
        distance_metric="cosine" if args.normalize_embeddings else "l2",
        quantize=args.quantize_index,
    )

    if args.pretrained_transformer:
        collate_fn = utils.process_with_esm_batch_converter(
            return_alignments=args.plot_recall_and_pid
        )
    elif args.msa_transformer:
        collate_fn = utils.msa_transformer_collate(just_sequences=True)
    else:
        collate_fn = non_default_collate

    # and create a test iterator.
    query_dataset = torch.utils.data.DataLoader(
        iterator,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
    )

    if args.compute_accuracy:
        compute_accuracy(
            query_dataset,
            index,
            rep_labels,
            model,
            n_neighbors=args.n_neighbors,
            pretrained_transformer=args.pretrained_transformer,
            normalize=args.normalize_embeddings,
            index_device=index_device,
            device=dev,
        )

    elif args.visualize:
        os.makedirs(args.image_path, exist_ok=True)
        utils.visualize_prediction_patterns(
            query_dataset=query_dataset,
            cluster_rep_index=index,
            cluster_rep_labels=rep_labels,
            cluster_rep_embeddings=rep_embeddings,
            cluster_rep_gapped_seqs=rep_gapped_seqs,
            trained_model=model,
            n_neighbors=args.n_neighbors,
            n_images=args.n_images,
            image_path=args.image_path,
            save_self_examples=args.save_self_examples,
            pretrained_transformer=args.pretrained_transformer,
            plot_dots=args.plot_dots,
            normalize=args.normalize_embeddings,
            index_device=index_device,
            device=dev,
        )
    elif args.plot_recall_and_pid:
        recall_at_pid_thresholds(
            query_dataset,
            index,
            rep_labels,
            model,
            n_neighbors=args.n_neighbors,
            pretrained_transformer=args.pretrained_transformer,
            index_device=index_device,
            device=dev,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    files = glob("/home/tc229954/data/prefilter/pfam/seed/20piddata/train/*afa")
    main(files)
