import pdb
from argparse import ArgumentParser
from sys import stdout

import esm
import matplotlib.pyplot as plt
import numpy as np
import torch

amino_n_to_a = [c for c in "ARNDCQEGHILKMFPSTWYVBZXJ*U"]
toks = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
]


def calculate_maximum_scoring_path(dot_products, image_filename):
    scores = torch.zeros_like(dot_products)
    scores[:, 0] = dot_products[:, 0]
    scores[0, :] = dot_products[0, :]

    best_path = torch.zeros((scores.shape[0], scores.shape[1], 2), dtype=torch.int)
    # for each row
    for i in range(1, scores.shape[0]):
        # for each column
        for j in range(1, scores.shape[1]):
            # where did we come from?
            # 0.2 as gap penalty
            vals = [scores[i - 1, j], scores[i, j - 1], scores[i - 1, j - 1]]
            idxs = [[i - 1, j], [i, j - 1], [i - 1, j - 1]]
            amax = np.argmax(vals)
            scores[i, j] = vals[amax] + dot_products[i, j]
            best_path[i, j] = torch.as_tensor(idxs[amax])

    # best column:
    best_col = torch.argmax(scores[-1, :])
    best_row = torch.argmax(scores[:, -1])
    if scores[-1, best_col] > scores[best_row, -1]:
        starting_point = best_path[-1, best_col]
    else:
        starting_point = best_path[best_row, -1]
    row_idx = starting_point[0]
    col_idx = starting_point[1]
    # while we haven't reached a side:
    path_log = [starting_point]
    total_score = 0

    while row_idx != 0 and col_idx != 0:
        next_best = best_path[row_idx, col_idx]
        total_score += dot_products[row_idx, col_idx]
        path_log.append(next_best)
        row_idx, col_idx = best_path[row_idx, col_idx]
    plt.imshow(dot_products, cmap="PiYG", vmin=0, vmax=1)
    for y, x in path_log:
        plt.scatter(x.item(), y.item(), c="r", s=0.5)

    match = "True" in image_filename
    plt.title(f"{match} match.")
    plt.colorbar()
    plt.savefig(f"{image_filename}")
    plt.close()


def _compute_count_mat_and_similarities(
    label_to_analyze,
    query_sequence,
    representative_embeddings,
    representative_labels,
    representative_index,
    index_device,
    pretrained_transformer,
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
    if pretrained_transformer:
        similarities = torch.matmul(
            query_sequence, representative_embedding.T.to(query_sequence.device)
        )
    else:
        similarities = torch.cdist(
            query_sequence, representative_embedding.to(query_sequence.device)
        )

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
            index_device,
            n_neighbors=n_neighbors,
        )
        # if there's a match to the predicted label
        for match_index in match_indices[0]:
            if representative_labels[match_index] == label_to_analyze:
                offset_index = match_index - representative_embedding_start_point
                count_mat[i, offset_index] += 1

    return count_mat, similarities


def most_common_matches(
    faiss_index,
    cluster_representative_labels,
    normalized_query_embedding,
    neighbors,
    device,
):
    distances, match_indices = search_index_device_aware(
        faiss_index,
        normalized_query_embedding.to(device),
        device,
        n_neighbors=neighbors,
    )
    if "cuda" in device:
        matches = cluster_representative_labels[match_indices.ravel().cpu().numpy()]
    else:
        matches = cluster_representative_labels[match_indices.ravel()]

    predicted_labels, counts = np.unique(matches, return_counts=True)

    return predicted_labels, counts


def search_index_device_aware(faiss_index, embedding, device, n_neighbors):
    if device == "cpu":
        distances, match_indices = faiss_index.search(
            embedding.to("cpu").numpy(), k=n_neighbors
        )
    else:
        distances, match_indices = faiss_index.search(embedding, k=n_neighbors)
    # strip dummy dimension
    return torch.as_tensor(distances), torch.as_tensor(match_indices)


@torch.no_grad()
def compute_cluster_representative_embeddings(
    representative_sequences,
    representative_labels,
    trained_model,
    normalize,
    device,
):
    # create representative tensor for raw sequences.
    # what, I'm not batching?
    representative_embeddings = []
    for rep_seq in representative_sequences:
        embed = (
            trained_model(rep_seq.unsqueeze(0).to(device))
            .transpose(-1, -2)
            .contiguous()
        )
        representative_embeddings.append(embed.squeeze())

    assert len(representative_embeddings) == len(representative_labels)
    # duplicate the labels the correct number of times
    _rep_labels = []
    for s, embed in zip(representative_labels, representative_embeddings):
        _rep_labels.extend([s] * embed.squeeze().shape[0])

    representative_labels = np.asarray(_rep_labels)
    representative_embeddings = torch.cat(representative_embeddings)
    if normalize:
        print("Normalizing...")
        representative_embeddings = torch.nn.functional.normalize(
            representative_embeddings, dim=-1
        )

    return representative_embeddings, representative_labels


def save_integer_encoded_sequences(filename, rep_seq, q_seq):
    with open(filename, "w") as dst:
        dst.write(">cluster representative\n")
        dst.write("".join([utils.amino_alphabet[i] for i in rep_seq]))
        dst.write("\n>query\n")
        dst.write("".join([utils.amino_alphabet[i] for i in q_seq]))


def save_string_sequences(filename, rep_seq, query_seq, true_seq=None):
    with open(filename, "w") as dst:
        dst.write(">cluster representative\n")
        dst.write(rep_seq)
        dst.write("\n>query\n")
        dst.write(query_seq)
        if true_seq is not None:
            dst.write("\n>true\n")
            dst.write(true_seq)


def create_parser():
    ap = ArgumentParser()
    ap.add_argument("model_root_dir")
    ap.add_argument("model_name")
    ap.add_argument("--include_all_families", action="store_true")
    ap.add_argument("--index_string", action="store_true")
    ap.add_argument("--compute_accuracy", action="store_true")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--index_device", type=str, default="cuda")
    ap.add_argument("--pretrained_transformer", action="store_true")
    ap.add_argument("--plot_dots", action="store_true")
    ap.add_argument("--dp_matrix", action="store_true")
    ap.add_argument("--daniel", action="store_true")
    ap.add_argument("--profmark", action="store_true")
    ap.add_argument("--normalize_embeddings", action="store_true")
    ap.add_argument("--msa_transformer", action="store_true")

    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--plot_recall_and_pid", action="store_true")
    ap.add_argument("--save_self_examples", action="store_true")
    ap.add_argument("--include_emission", action="store_true")
    ap.add_argument("--n_seq_per_target_family", type=int)
    ap.add_argument("--image_path", type=str, default="debug")
    ap.add_argument("--n_neighbors", type=int, default=10)
    ap.add_argument("--n_images", type=int, default=10)

    return ap


def calculate_pid(s1, s2):
    if not isinstance(s2, list):
        s2 = [s2]
    # get maximum percent identity
    total_pid = 0
    for sequence in s2:
        numerator = 0
        denominator = 0
        for res1, res2 in zip(s1, sequence):
            if res1 == "." and res2 == ".":
                continue
            if res1 == res2:
                numerator += 1
            else:
                denominator += 1
        total_pid += numerator / denominator

    return total_pid / len(s2)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def recall_at_pid_thresholds(
    query_dataset,
    cluster_rep_index,
    cluster_rep_labels,
    trained_model,
    n_neighbors,
    pretrained_transformer,
    index_device,
    device="cuda",
):
    # going to store the alignments in the query dataset.
    total_sequences = 0
    if pretrained_transformer:
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

    pid_to_recall = defaultdict(int)
    pid_to_total = defaultdict(int)

    for j, (features, alis, labels, sequences) in enumerate(query_dataset):
        if pretrained_transformer:
            embeddings = trained_model(
                features.to(device), repr_layers=[33], return_contacts=False
            )

            embed = []
            for k, seq in enumerate(sequences):
                embed.append(embeddings["representations"][33][k, 1 : len(seq) + 1])
            embeddings = embed
        else:
            embeddings = trained_model(features.to(device)).transpose(-1, -2)

        stdout.write(f"{j / len(query_dataset):.3f}\r")

        for label, sequence, gapped_sequence in zip(labels, embeddings, alis):
            total_sequences += 1
            sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()
            train_alignment = query_dataset.dataset.label_to_seed_alignment[label]
            pid = calculate_pid(gapped_sequence, train_alignment)
            predicted_labels, counts = most_common_matches(
                cluster_rep_index,
                cluster_rep_labels,
                sequence,
                n_neighbors,
                index_device,
            )
            # get percent identity;
            # return
            top_preds = predicted_labels[np.argsort(counts)]
            if label == top_preds[-1]:
                pid_to_recall[int(100 * pid)] += 1
            pid_to_total[int(100 * pid)] += 1

    total = np.zeros(len(pid_to_total))
    recall = np.zeros(len(pid_to_recall))
    for i, key in enumerate(sorted(pid_to_recall.keys())):
        total[i] = pid_to_total[key]
        recall[i] = pid_to_recall[key]
    # do the aggregation here.
    plt.plot(
        moving_average(sorted(pid_to_recall.keys())),
        moving_average(recall) / moving_average(total),
    )
    plt.savefig("test256.png")
    plt.close()


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
    plot_dots,
    index_device,
    normalize,
    device="cuda",
):
    if pretrained_transformer:
        _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

    if plot_dots:
        print("Plotting dot products of top matches to each query sequence.")

    image_idx = 0
    for features, labels, gapped_sequences in query_dataset:
        if pretrained_transformer:
            embeddings = trained_model(
                features.to(device), repr_layers=[33], return_contacts=False
            )

            embed = []
            for k, seq in enumerate(gapped_sequences):
                embed.append(embeddings["representations"][33][k, 1 : len(seq) + 1])
            embeddings = embed
        else:
            embeddings = trained_model(features.to(device)).transpose(-1, -2)

        for feat_idx, (label, sequence) in enumerate(zip(labels, embeddings)):
            if normalize:
                sequence = torch.nn.functional.normalize(sequence, dim=-1).contiguous()
            else:
                sequence = sequence.contiguous()

            predicted_labels, counts = most_common_matches(
                cluster_rep_index,
                cluster_rep_labels,
                sequence,
                n_neighbors,
                index_device,
            )
            predicted_labels = predicted_labels[np.argsort(counts)]
            # now, get the dp matrix going.
            top_pred_embedding = cluster_rep_embeddings[
                cluster_rep_labels == predicted_labels[-1]
            ]
            dots = torch.matmul(top_pred_embedding.to(sequence.device), sequence.T)
            top_pred_seq = cluster_rep_gapped_seqs[predicted_labels[-1]].replace(
                ".", ""
            )
            if pretrained_transformer:
                query_gapped_seq = gapped_sequences[feat_idx].replace(".", "")
            else:
                query_gapped_seq = ""

            jj = image_idx

            fpath = f"transformer/transformer_{predicted_labels[-1] == label}_{jj}"
            calculate_maximum_scoring_path(
                dots.to("cpu"), image_filename=fpath + ".png"
            )
            with open(f"query_{jj}.fa", "w") as dst:
                dst.write(">query\n")
                dst.write(f"{query_gapped_seq}\n")
            with open(f"top_match_{jj}.fa", "w") as dst:
                dst.write(">top_match\n")
                dst.write(f"{top_pred_seq}\n")
            image_idx += 1
            continue

            if plot_dots:
                # i want a bunch of false matches and then the true one.
                fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(13, 10))
                for kk, lab in enumerate(predicted_labels[::-1][:10]):
                    rep_embedding = cluster_rep_embeddings[cluster_rep_labels == lab]
                    l2s = torch.matmul(sequence, rep_embedding.to(sequence.device).T)
                    ax[kk % 5, kk // 5].imshow(
                        l2s.to("cpu"),
                        cmap="PiYG",
                    )
                    ax[kk % 5, kk // 5].set_title(
                        f"{kk + 1} ranked match. {lab == label} match."
                    )
                    ax[kk % 5, kk // 5].set_xticks([])
                    ax[kk % 5, kk // 5].set_yticks([])

                rep_embedding = cluster_rep_embeddings[cluster_rep_labels == label]

                l2s = torch.matmul(sequence, rep_embedding.to(sequence.device).T)

                ax[kk % 5, kk // 5].imshow(
                    l2s.to("cpu"),
                    cmap="PiYG",
                )
                ax[kk % 5, kk // 5].set_xticks([])
                ax[kk % 5, kk // 5].set_yticks([])

                real_rank = np.argwhere(predicted_labels[::-1] == label)

                if not real_rank.shape[0]:
                    ax[kk % 5, kk // 5].set_title(
                        f"No true hits in {predicted_labels.shape} matches."
                    )
                else:
                    ax[kk % 5, kk // 5].set_title(
                        f"True hit showed up in the {real_rank[0][0] + 1}th/nd/st most common match."
                    )

                if predicted_labels[-1] == label:
                    plt.savefig(
                        f"{image_path}/dots_true_hit_{image_idx}.png",
                        bbox_inches="tight",
                    )
                else:
                    plt.savefig(
                        f"{image_path}/dots_false_hit_{image_idx}.png",
                        bbox_inches="tight",
                    )

                image_idx += 1

                plt.close()

            predicted_label = predicted_labels[-1]
            second_label = predicted_labels[-2]
            representative_gapped_seq = cluster_rep_gapped_seqs[predicted_label]
            query_gapped_seq = ""

            first_cmat, first_sim = _compute_count_mat_and_similarities(
                predicted_label,
                sequence,
                cluster_rep_embeddings,
                cluster_rep_labels,
                cluster_rep_index,
                index_device,
                pretrained_transformer=pretrained_transformer,
                n_neighbors=n_neighbors,
            )

            second_cmat, second_sim = _compute_count_mat_and_similarities(
                second_label,
                sequence,
                cluster_rep_embeddings,
                cluster_rep_labels,
                cluster_rep_index,
                index_device,
                pretrained_transformer=pretrained_transformer,
                n_neighbors=n_neighbors,
            )
            unique_name = f"{n_neighbors}_neigh_{image_idx}"

            if save_self_examples:
                self_sim = torch.matmul(sequence, sequence.T)
                if pretrained_transformer:
                    plt.imshow(self_sim.to("cpu"), cmap="PiYG")
                else:
                    plt.imshow(self_sim.to("cpu"))
                plt.title("self-similarity")
                plt.colorbar()
                plt.savefig(
                    f"{image_path}/no_mlp_self_{unique_name}.png", bbox_inches="tight"
                )
                plt.close()

            if predicted_label == label:

                fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(13, 10))
                if pretrained_transformer:
                    ax[0, 0].imshow(first_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[0, 0].imshow(first_sim.to("cpu"), cmap="PiYG")
                ax[0, 0].set_title("dots")

                ax[0, 1].set_title(
                    f"true hit. n hits to sequence: {np.sum(first_cmat):.3f}"
                )
                ax[0, 1].imshow(first_cmat)
                if pretrained_transformer:
                    ax[1, 0].imshow(second_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[1, 0].imshow(second_sim.to("cpu"), cmap="PiYG")
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
                    ax[0, 0].imshow(first_sim.to("cpu"), cmap="PiYG")
                ax[0, 0].set_title("dots")

                ax[0, 1].set_title(
                    f"false hit, rank1. n hits to sequence: {np.sum(first_cmat):.3f}"
                )
                ax[0, 1].imshow(first_cmat)

                if pretrained_transformer:
                    ax[1, 0].imshow(second_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[1, 0].imshow(second_sim.to("cpu"), cmap="PiYG")

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
                    index_device,
                    pretrained_transformer=pretrained_transformer,
                    n_neighbors=n_neighbors,
                )
                if pretrained_transformer:
                    ax[2, 0].imshow(true_sim.to("cpu"), cmap="PiYG")
                else:
                    ax[2, 0].imshow(true_sim.to("cpu"), cmap="PiYG")
                ax[2, 0].set_title("dots")
                ax[2, 1].set_title(
                    f"hit on true match. n hits to sequence: {np.sum(true_cmat):.3f}"
                )
                ax[2, 1].imshow(true_cmat)
                plt.savefig(
                    f"{image_path}/false_{unique_name}.png", bbox_inches="tight"
                )

                true_gapped_seq = cluster_rep_gapped_seqs[label]

                save_string_sequences(
                    f"{image_path}/false_{unique_name}.fa",
                    representative_gapped_seq,
                    query_gapped_seq,
                    true_seq=true_gapped_seq,
                )

            image_idx += 1
            plt.close()

            if (image_idx + 1) == n_images:
                exit()
