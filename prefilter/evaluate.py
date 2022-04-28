import pdb
from glob import glob
from collections import defaultdict
from sys import stdout

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import yaml
from argparse import ArgumentParser
from typing import List
import prefilter.utils as utils


def create_parser():
    ap = ArgumentParser()
    ap.add_argument("model_root_dir")
    ap.add_argument("model_name")
    ap.add_argument("--clustered_split", action="store_true")
    ap.add_argument("--compute_accuracy", action="store_true")
    ap.add_argument("--embed_dim", type=int, default=256)

    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--n_seq_per_target_family", type=int)
    ap.add_argument("--image_path", type=str, default="debug")
    ap.add_argument("--n_neighbors", type=int, default=10)
    ap.add_argument("--n_images", type=int, default=10)

    return ap


def non_default_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([torch.tensor(b[1]) for b in batch]),
        [b[2] for b in batch],
    )


def save_integer_encoded_sequences(filename, rep_seq, q_seq):
    with open(filename, "w") as dst:
        dst.write(">cluster representative\n")
        dst.write("".join([utils.amino_alphabet[i] for i in rep_seq]))
        dst.write("\n>query\n")
        dst.write("".join([utils.amino_alphabet[i] for i in q_seq]))


def save_string_sequences(filename, rep_seq, query_seq):
    with open(filename, "w") as dst:
        dst.write(">cluster representative\n")
        dst.write(rep_seq)
        dst.write("\n>query\n")
        dst.write(query_seq)


def compute_cluster_representative_embeddings(
    representative_sequences, representative_labels, trained_model, device
):
    """
    :param representative_sequences: Cluster reps.
    :type representative_sequences: List of lists of integer encodings of sequences.
    :param representative_labels: List of cluster rep labels.
    :type representative_labels:
    :param trained_model: A trained neural network.
    :type trained_model: torch.nn.Module.
    :return: embeddings, labels
    :rtype:
    """
    representative_tensor = torch.stack(representative_sequences).to(device)
    representative_embeddings = (
        trained_model(representative_tensor).transpose(-1, -2).contiguous()
    )
    # duplicate the labels the correct number of times
    _rep_labels = []
    for s, embed in zip(representative_labels, representative_embeddings):
        _rep_labels.extend([s] * embed.shape[0])
    representative_labels = np.asarray(_rep_labels)
    representative_embeddings = torch.cat(
        torch.unbind(representative_embeddings, axis=0)
    )
    representative_embeddings = torch.nn.functional.normalize(
        representative_embeddings, dim=-1
    )
    assert representative_labels.shape[0] == representative_embeddings.shape[0]
    return representative_embeddings, representative_labels


def search_index_device_aware(faiss_index, embedding, device, n_neighbors):
    if device == "cpu":
        distances, match_indices = faiss_index.search(embedding.numpy(), k=n_neighbors)
    else:
        distances, match_indices = faiss_index.search(embedding, k=n_neighbors)
    # strip dummy dimension
    return distances, match_indices


def most_common_matches(
    faiss_index,
    cluster_representative_labels,
    normalized_query_embedding,
    neighbors,
    device,
):
    distances, match_indices = search_index_device_aware(
        faiss_index, normalized_query_embedding, device, n_neighbors=neighbors
    )
    if device == "cuda":
        matches = cluster_representative_labels[match_indices.ravel().cpu().numpy()]
    else:
        matches = cluster_representative_labels[match_indices.ravel()]
    predicted_labels, counts = np.unique(matches, return_counts=True)

    return predicted_labels, counts


def compute_accuracy(
    query_dataset,
    cluster_rep_index,
    cluster_rep_labels,
    trained_model,
    n_neighbors,
    device="cuda",
):
    total_sequences = 0
    thresholds = [1, 2, 3, 5, 10, 20, 100, 200]
    topn = defaultdict(int)

    for j, (features, labels, _) in enumerate(query_dataset):
        embeddings = trained_model(features.to(device)).transpose(-1, -2)
        stdout.write(f"{j / len(query_dataset):.3f}\r")
        for label, sequence in zip(labels, embeddings):

            total_sequences += 1
            sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()

            predicted_labels, counts = most_common_matches(
                cluster_rep_index, cluster_rep_labels, sequence, n_neighbors, device
            )

            top_preds = predicted_labels[np.argsort(counts)]

            for n in thresholds:
                top_pred = top_preds[-n:]
                if label.item() in set(top_pred):
                    topn[n] += 1

    correct_counts = np.asarray([topn[t] for t in thresholds])
    thresholds = ", ".join([str(t) for t in thresholds])
    percent_correct = ", ".join([f"{c / total_sequences:.3f}" for c in correct_counts])
    _, total_families = np.unique(cluster_rep_labels, return_counts=True)

    print(
        f"{thresholds}\n",
        f"{percent_correct}\n"
        f"percent of sequences where the correct family was the nth most common match. "
        f"Total families searched: {len(total_families)}",
        f"Total sequences: {total_sequences}",
    )


def visualize_prediction_patterns(
    query_dataset,
    cluster_rep_index,
    cluster_rep_labels,
    cluster_rep_embeddings,
    cluster_rep_gapped_seqs,
    trained_model,
    n_neighbors,
    n_images,
    image_path,
    device="cuda",
):
    image_idx = 0
    for features, labels, gapped_sequences in query_dataset:
        embeddings = trained_model(features.to(device)).transpose(-1, -2)
        for feat_idx, (label, sequence) in enumerate(zip(labels, embeddings)):
            sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()

            predicted_labels, counts = most_common_matches(
                cluster_rep_index, cluster_rep_labels, sequence, n_neighbors, device
            )

            predicted_labels = predicted_labels[np.argsort(counts)]
            # random sampling. I'll write the sequences into
            # a file with the same name as the image. Then looking up their
            # alignments will be a simple _grep_.
            # grab the most common label (isn't always correct).
            predicted_label = predicted_labels[-1]
            true_label = label
            n_hits = sorted(counts)[-1]
            # grab the representative embedding
            representative_embedding = cluster_rep_embeddings[
                cluster_rep_labels == predicted_label
            ]
            true_rep_embedding = cluster_rep_embeddings[
                cluster_rep_labels == true_label.item()
            ]
            # we index it at label because we constructed the representative
            # matrix in a for loop, so each sequences label is its position in the list.
            representative_gapped_seq = cluster_rep_gapped_seqs[predicted_label]
            true_gapped_seq = cluster_rep_gapped_seqs[true_label]

            representative_embedding_start_point = np.where(
                cluster_rep_labels == predicted_label
            )[0][0]

            true_representative_embedding_start_point = np.where(
                cluster_rep_labels == true_label.item()
            )[0][0]

            query_gapped_seq = gapped_sequences[feat_idx]

            similarities = torch.matmul(sequence, representative_embedding.T)
            true_similarities = torch.matmul(sequence, true_rep_embedding.T)

            count_mat = np.zeros((sequence.shape[0], representative_embedding.shape[0]))
            true_count_mat = np.zeros((sequence.shape[0], true_rep_embedding.shape[0]))

            for i, amino_acid in enumerate(sequence):
                distances, match_indices = search_index_device_aware(
                    cluster_rep_index,
                    amino_acid.unsqueeze(0),
                    device,
                    n_neighbors=n_neighbors,
                )
                # if there's a match to the predicted label
                for match_index in match_indices[0]:
                    if cluster_rep_labels[match_index] == predicted_label:
                        offset_index = (
                            match_index - representative_embedding_start_point
                        )
                        count_mat[i, offset_index] += 1

            for i, amino_acid in enumerate(sequence):
                distances, match_indices = search_index_device_aware(
                    cluster_rep_index,
                    amino_acid.unsqueeze(0),
                    device,
                    n_neighbors=n_neighbors,
                )
                # if there's a match to the predicted label
                for match_index in match_indices[0]:
                    if cluster_rep_labels[match_index] == true_label:
                        offset_index = (
                            match_index - true_representative_embedding_start_point
                        )
                        true_count_mat[i, offset_index] += 1

            # extrememly verbose plotting code :(
            fig, ax = plt.subplots(ncols=4, figsize=(13, 10))
            ax[0].imshow(count_mat)
            ax[0].set_title(f"n hits to sequence: {n_hits}")

            ax[2].imshow(true_count_mat)
            ax[2].set_title(f"true n hits to sequence: {np.sum(true_count_mat)}")

            ax[3].imshow(true_similarities.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")
            ax[3].set_title("true sim.")

            sim_ax = ax[1].imshow(similarities.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")
            ax[1].set_title("dot products")
            # set up colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(sim_ax, cax=cbar_ax)

            unique_name = f"{n_neighbors}_neigh_{image_idx}"

            if predicted_label == label:
                print("saving true")
                plt.suptitle("true match")
                plt.savefig(f"{image_path}/true_{unique_name}.png", bbox_inches="tight")
                save_string_sequences(
                    f"{image_path}/true_{unique_name}.fa",
                    representative_gapped_seq,
                    query_gapped_seq,
                )
            else:
                print("saving false")
                plt.suptitle("false match")
                plt.savefig(
                    f"{image_path}/false_{unique_name}.png", bbox_inches="tight"
                )
                save_string_sequences(
                    f"{image_path}/fp_true_{unique_name}.fa",
                    true_gapped_seq,
                    query_gapped_seq,
                )
                # save the representative sequence.
                # and the query sequence.
                save_string_sequences(
                    f"{image_path}/false_{unique_name}.fa",
                    representative_gapped_seq,
                    query_gapped_seq,
                )

            plt.close()

            image_idx += 1
            if (image_idx + 1) == n_images:
                exit()


@torch.no_grad()
def main(fasta_files, min_seq_len=256, batch_size=32):
    parser = create_parser()
    args = parser.parse_args()

    # get model files
    hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
    model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
    embed_dim = args.embed_dim

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with open(hparams_path, "r") as src:
        hparams = yaml.safe_load(src)

    # Set up model and dataset
    if not os.path.isfile(model_path):
        raise ValueError(f"No model found at {model_path}")

    model, _ = utils.load_model(model_path, hparams, dev)
    iterator = utils.ClusterIterator(
        fasta_files,
        min_seq_len,
        representative_index=0,
        evaluate_on_clustered_split=args.clustered_split,
        n_seq_per_target_family=args.n_seq_per_target_family,
    )

    rep_seqs, rep_labels = iterator.get_cluster_representatives()
    rep_gapped_seqs = iterator.seed_gapped_sequences
    # stack the seed sequences.
    rep_embeddings, rep_labels = compute_cluster_representative_embeddings(
        rep_seqs, rep_labels, model, device=dev
    )
    # create an index
    index = utils.create_faiss_index(rep_embeddings, embed_dim, device=dev)

    # and create a test iterator.
    query_dataset = torch.utils.data.DataLoader(
        iterator, batch_size=batch_size, collate_fn=non_default_collate, shuffle=True
    )

    if args.compute_accuracy:
        compute_accuracy(query_dataset, index, rep_labels, model, 10, dev)
    elif args.visualize:
        os.makedirs(args.image_path, exist_ok=True)
        visualize_prediction_patterns(
            query_dataset,
            index,
            rep_labels,
            rep_embeddings,
            rep_gapped_seqs,
            model,
            args.n_neighbors,
            args.n_images,
            args.image_path,
            dev,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    pids = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
    pids = [0.8]
    for pid in sorted(pids):
        pfam_files = glob(
            f"/home/tc229954/data/prefilter/pfam/seed/clustered/{pid}/*-train.fa"
        )
        if len(pfam_files):
            print(f"pid: {1 - pid}")
            main(pfam_files)
            print("=")
