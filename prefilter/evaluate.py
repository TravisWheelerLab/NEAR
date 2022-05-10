import pdb
from glob import glob
from collections import defaultdict
from sys import stdout

import faiss
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import yaml
from argparse import ArgumentParser
import esm
from typing import List
import prefilter.utils as utils

from pytorch_lightning import seed_everything


def create_parser():
    ap = ArgumentParser()
    ap.add_argument("model_root_dir")
    ap.add_argument("model_name")
    ap.add_argument("--clustered_split", action="store_true")
    ap.add_argument("--compute_accuracy", action="store_true")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--pretrained_transformer", action="store_true")

    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--save_self_examples", action="store_true")
    ap.add_argument("--include_emission", action="store_true")
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


@torch.no_grad()
def compute_cluster_representative_embeddings(
    representative_sequences,
    representative_labels,
    trained_model,
    device,
    pretrained_transformer,
):
    """
    :param pretrained_transformer: bool
    :type pretrained_transformer: bool
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

    if pretrained_transformer:
        batch_size = 64
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        trained_model = trained_model.to(device)
        batch_converter = alphabet.get_batch_converter()
        # batch_labels, batch_strs, batch_tokens = batch_converter(data)
        data = []
        for j, prot_seq in enumerate(representative_tensor):
            data.append(
                (
                    f"prot_{j}",
                    "".join([utils.amino_alphabet[i.item()] for i in prot_seq]),
                )
            )

        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        representative_embeddings = []

        for i in range(0, len(batch_tokens) - batch_size, batch_size):
            stdout.write(f"{i / len(batch_tokens):.3f}\r")
            embeddings = trained_model(
                batch_tokens[i : i + batch_size].to(device),
                repr_layers=[33],
                return_contacts=False,
            )
            # send to cpu so we save on some GPU memory.
            # large embedding dim
            representative_embeddings.append(
                embeddings["representations"][33][:, 1:-1].detach().to("cpu")
            )
        print()
        representative_embeddings = torch.cat(representative_embeddings, dim=0)

    else:
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
    if not pretrained_transformer:
        representative_embeddings = torch.nn.functional.normalize(
            representative_embeddings, dim=-1
        )
    print("normalizing.")
    representative_embeddings = torch.nn.functional.normalize(
        representative_embeddings, dim=-1
    )
    # assert len(representative_labels) == len(representative_embeddings)
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
    pretrained_transformer,
    device="cuda",
):
    total_sequences = 0
    thresholds = [1, 2, 3, 5, 10, 20, 100, 200]
    topn = defaultdict(int)

    if pretrained_transformer:
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()

    for j, (features, labels, _) in enumerate(query_dataset):
        if pretrained_transformer:
            embeddings = _infer_with_transformer(
                features, trained_model, batch_converter, device
            )
        else:
            embeddings = trained_model(features.to(device)).transpose(-1, -2)

        stdout.write(f"{j / len(query_dataset):.3f}\r")

        # searching each sequence separately against the index is probably slow.
        for label, sequence in zip(labels, embeddings):

            total_sequences += 1

            if not pretrained_transformer:
                sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()
                predicted_labels, counts = most_common_matches(
                    cluster_rep_index, cluster_rep_labels, sequence, n_neighbors, device
                )
            else:
                sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()
                # predicted_labels, counts = most_common_matches(
                sequence = sequence.contiguous()
                # sequence = sequence.to("cpu")
                predicted_labels, counts = most_common_matches(
                    cluster_rep_index,
                    cluster_rep_labels,
                    sequence,
                    n_neighbors,
                    device=device,
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
        f"{percent_correct}\n" f"Total families searched: {len(total_families)}",
        f"Total sequences: {total_sequences}",
    )


def _compute_count_mat_and_similarities(
    label_to_analyze,
    query_sequence,
    representative_embeddings,
    representative_labels,
    representative_index,
    device,
    n_neighbors,
):
    """
    Gets matrices of dot prods and counts. Label_to_analyze is used to look
    up the embedding of the sequence that our query sequence matched to.
    This function grabs the embedding associated with the label to analyze
    and gets its dots. And it grabs the number of times the label to analyze showed
    up in each cell of the dots b/t the LtA sequence and the query sequence (count_mat).
    """
    if isinstance(label_to_analyze, torch.Tensor):
        label_to_analyze = label_to_analyze.item()

    representative_embedding = representative_embeddings[
        representative_labels == label_to_analyze
    ]
    representative_embedding_start_point = np.where(
        representative_labels == label_to_analyze
    )
    similarities = torch.matmul(query_sequence, representative_embedding.T)
    count_mat = np.zeros((query_sequence.shape[0], representative_embedding.shape[0]))
    if len(representative_embedding_start_point):
        representative_embedding_start_point = representative_embedding_start_point[0][
            0
        ]
    else:
        pdb.set_trace()
        return count_mat, similarities

    for i, amino_acid in enumerate(query_sequence):
        distances, match_indices = search_index_device_aware(
            representative_index,
            amino_acid.unsqueeze(0),
            device,
            n_neighbors=n_neighbors,
        )
        # if there's a match to the predicted label
        for match_index in match_indices[0]:
            if representative_labels[match_index] == label_to_analyze:
                offset_index = match_index - representative_embedding_start_point
                count_mat[i, offset_index] += 1

    return count_mat, similarities


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
    save_self_examples,
    pretrained_transformer,
    device="cuda",
):
    if pretrained_transformer:
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()

    image_idx = 0
    for features, labels, gapped_sequences in query_dataset:
        if pretrained_transformer:
            embeddings = _infer_with_transformer(
                features, trained_model, batch_converter, device
            )
        else:
            embeddings = trained_model(features.to(device)).transpose(-1, -2)

        for feat_idx, (label, sequence) in enumerate(zip(labels, embeddings)):
            if pretrained_transformer:
                sequence = sequence.contiguous().to("cpu")
            else:
                sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()

            # searching the index takes a huge amount of time.
            predicted_labels, counts = most_common_matches(
                cluster_rep_index, cluster_rep_labels, sequence, n_neighbors, device
            )

            predicted_labels = predicted_labels[np.argsort(counts)]
            predicted_label = predicted_labels[-1]
            second_label = predicted_labels[-2]

            representative_gapped_seq = cluster_rep_gapped_seqs[predicted_label]
            query_gapped_seq = "".join(
                utils.amino_alphabet[i.item()] for i in features[feat_idx]
            )

            first_cmat, first_sim = _compute_count_mat_and_similarities(
                predicted_label,
                sequence,
                cluster_rep_embeddings,
                cluster_rep_labels,
                cluster_rep_index,
                device,
                n_neighbors,
            )

            second_cmat, second_sim = _compute_count_mat_and_similarities(
                second_label,
                sequence,
                cluster_rep_embeddings,
                cluster_rep_labels,
                cluster_rep_index,
                device,
                n_neighbors,
            )
            unique_name = f"{n_neighbors}_neigh_{image_idx}"

            if save_self_examples:
                self_sim = torch.matmul(sequence, sequence.T)
                if pretrained_transformer:
                    plt.imshow(self_sim.to("cpu"), cmap="PiYG")
                else:
                    plt.imshow(self_sim.to("cpu"), vmin=-1, vmax=1)
                plt.title("self-similarity")
                plt.colorbar()
                plt.savefig(
                    f"{image_path}/no_mlp_self_{unique_name}.png", bbox_inches="tight"
                )
                plt.close()

            if predicted_label == label:

                fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13, 10))
                if pretrained_transformer:
                    print(first_sim.shape)
                    ax[0, 0].imshow(first_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[0, 0].imshow(first_sim.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")
                ax[0, 0].set_title("dots")

                ax[0, 1].set_title(
                    f"true hit. n hits to sequence: {np.sum(first_cmat):.3f}"
                )
                ax[0, 1].imshow(first_cmat)
                if pretrained_transformer:
                    ax[1, 0].imshow(second_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[1, 0].imshow(second_sim.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")
                ax[1, 0].set_title("dots")

                ax[1, 1].set_title(
                    f"second hit. n hits to sequence: {np.sum(second_cmat):.3f}"
                )
                ax[1, 1].imshow(second_cmat)

                plt.savefig(f"{image_path}/true_{unique_name}.png", bbox_inches="tight")
                save_string_sequences(
                    f"{image_path}/true_{unique_name}.fa",
                    representative_gapped_seq,
                    query_gapped_seq,
                )
                # two different imshows - first and second matches.
            else:
                # three different imshows
                # first, compare to the true label (then to the first and second
                # matches
                fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(13, 10))
                if pretrained_transformer:
                    ax[0, 0].imshow(first_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[0, 0].imshow(first_sim.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")
                ax[0, 0].set_title("dots")

                ax[0, 1].set_title(
                    f"false hit, rank1. n hits to sequence: {np.sum(first_cmat):.3f}"
                )
                ax[0, 1].imshow(first_cmat)

                if pretrained_transformer:
                    ax[1, 0].imshow(second_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[1, 0].imshow(second_sim.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")

                ax[1, 0].set_title("dots")

                if second_label != label:
                    ax[1, 1].set_title(
                        f"false hit, rank2. n hits to sequence: {np.sum(second_cmat):.3f}"
                    )
                else:
                    ax[1, 1].set_title(
                        f"true hit, rank2. n hits to sequence: {np.sum(second_cmat):.3f}"
                    )

                ax[1, 1].imshow(second_cmat)

                true_cmat, true_sim = _compute_count_mat_and_similarities(
                    label,
                    sequence,
                    cluster_rep_embeddings,
                    cluster_rep_labels,
                    cluster_rep_index,
                    device,
                    n_neighbors,
                )
                if pretrained_transformer:
                    ax[2, 0].imshow(true_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[2, 0].imshow(true_sim.to("cpu"), vmin=-1, vmax=1, cmap="PiYG")
                ax[2, 0].set_title("dots")
                ax[2, 1].set_title(
                    f"hit on true match. n hits to sequence: {np.sum(true_cmat):.3f}"
                )
                ax[2, 1].imshow(true_cmat)
                plt.savefig(
                    f"{image_path}/false_{unique_name}.png", bbox_inches="tight"
                )

                save_string_sequences(
                    f"{image_path}/false_{unique_name}.fa",
                    representative_gapped_seq,
                    query_gapped_seq,
                )

            image_idx += 1
            plt.close()

            if (image_idx + 1) == n_images:
                exit()


def _infer_with_transformer(seq_batch, trained_model, batch_converter, device):
    data = []

    for j, prot_seq in enumerate(seq_batch):
        data.append(
            (f"prot_{j}", "".join([utils.amino_alphabet[i.item()] for i in prot_seq]))
        )
    _, _, batch_tokens = batch_converter(data)

    embeddings = trained_model(
        batch_tokens.to(device), repr_layers=[33], return_contacts=False
    )["representations"][33][:, 1:-1]

    return embeddings.to(device)


@torch.no_grad()
def main(fasta_files, min_seq_len=256, batch_size=32):
    parser = create_parser()
    args = parser.parse_args()

    embed_dim = args.embed_dim
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # get model files
    if args.pretrained_transformer:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results
        embed_dim = 1280
    else:
        hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
        model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)

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
        rep_seqs,
        rep_labels,
        model,
        device=dev,
        pretrained_transformer=args.pretrained_transformer,
    )
    # create an index
    if args.pretrained_transformer:
        index = utils.create_faiss_index(
            rep_embeddings, embed_dim, device=dev, distance_metric="cosine"
        )
    else:
        index = utils.create_faiss_index(rep_embeddings, embed_dim, device=dev)

    # and create a test iterator.
    query_dataset = torch.utils.data.DataLoader(
        iterator, batch_size=batch_size, collate_fn=non_default_collate, shuffle=True
    )

    if args.compute_accuracy:
        compute_accuracy(
            query_dataset,
            index,
            rep_labels,
            model,
            n_neighbors=args.n_neighbors,
            pretrained_transformer=args.pretrained_transformer,
            device=dev,
        )

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
            args.save_self_examples,
            args.pretrained_transformer,
            dev,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    # pids = [0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.7, 0.75, 0.8, 0.9]
    pids = [0.8]

    for pid in sorted(pids):
        pfam_files = glob(
            f"/home/tc229954/data/prefilter/pfam/seed/clustered/{pid}/*-train.fa"
        )
        if len(pfam_files):
            print(f"pid: {1 - pid:.3f}")
            main(pfam_files)
            print("=")
