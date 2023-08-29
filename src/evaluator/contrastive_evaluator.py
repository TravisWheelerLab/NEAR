"""
Final version of the evaluator class. 
This is preferred when using multiprocessing """

import logging
import os
from typing import List, Tuple
import tqdm
import faiss
import numpy as np
import torch
import time
from src.utils import create_faiss_index, encode_string_sequence
from collections import defaultdict

logger = logging.getLogger("evaluate")


def filter_scores(scores_array, indices_array, unrolled_names):
    """Filters the scores such that every query amino can only
    be matched to one amino from each target sequence
    and it matches the one with the biggest score.

    Then sums the scores that belong to the same target
    and returns the resulting distances in a dict

    scores_array (numqueryaminos, 1000): an array of 1000 scores per query amino
    indices_array: (numqueryaminos, 1000) the indices of the target sequence name (in unrolled_names)
    for each of the scores in scores_array"""

    filtered_scores = defaultdict(float)

    # iterate over query amino scores
    for match_idx in range(len(scores_array)):
        match_scores = scores_array[match_idx]
        names = unrolled_names[
            indices_array[match_idx]
        ]  # the names of the targets for each 1000 hits
        sorted_match_idx = np.argsort(match_scores)[::-1]

        _, unique_indices = np.unique(
            names[sorted_match_idx], return_index=True
        )  # the unique names of the targets for each 1000 hits (<= 1000)

        new_indices = list(indices_array[match_idx][sorted_match_idx][unique_indices])
        new_scores = list(match_scores[sorted_match_idx][unique_indices])

        for distance, name in zip(new_scores, unrolled_names[new_indices]):
            filtered_scores[name] += distance

    return filtered_scores


def filter_and_calc_embeddings(
    names: List[str],
    sequences: List[str],
    model_class,
    max_seq_length=512,
    model_device="cpu",
    minimum_seq_length=0,
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """Filters the sequences by length thresholding given the
    minimum and maximum length threshold variables"""
    embeddings = []
    lengths = []

    filtered_names = names.copy()
    num_removed = 0
    print("Embedding queries...")
    for name, sequence in zip(names, sequences):
        length = len(sequence)
        if max_seq_length >= length >= minimum_seq_length:
            embed = (
                model_class(
                    encode_string_sequence(sequence).unsqueeze(0).to(model_device)
                )
                .squeeze()
                .T
            )
            # return: seq_lenxembed_dim shape
            embeddings.append(torch.nn.functional.normalize(embed, dim=-1).to("cpu"))
            lengths.append(length)
        else:
            num_removed += 1
            filtered_names.remove(name)
            # filtered_sequences.remove(sequence)
    return filtered_names, embeddings, lengths


@torch.no_grad()
def _calc_embeddings(
    sequences,
    model_class,
    model_device="cpu",
    max_seq_length=512,
):
    embeddings = []
    lengths = []
    indices = []

    for idx, sequence in enumerate(sequences):
        if len(sequence) < max_seq_length:
            embed = (
                model_class(
                    encode_string_sequence(sequence).unsqueeze(0).to(model_device)
                )
                .squeeze()
                .T
            )
            embeddings.append(torch.nn.functional.normalize(embed, dim=-1).to("cpu"))
            lengths.apend(len(sequence))
            indices.append(idx)
    return embeddings, lengths, indices


def reduce_indices(indices, index_mapping):
    new_indices = np.zeros_like(indices)

    for i, amino_index_list in enumerate(indices):
        new_indices[i] = np.array([index_mapping[idx] for idx in amino_index_list])

    return new_indices


def search(args):
    (query_data, model, index_mapping, output_path, index, max_seq_length) = args
    query_names, queries, _ = _calc_embeddings(query_data, model, max_seq_length)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    all_scores = []
    all_indices = []

    print("Searching...")
    for i in tqdm.tqdm(range(len(queries))):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        all_scores.append(scores.to("cpu").numpy())

        #   all_indices.append(reduce_indices(indices.to("cpu").numpy(), index_mapping))
        all_indices.append(indices.to("cpu").numpy())
    return query_names, all_scores, all_indices


@torch.no_grad()
def search_and_filter(args):
    (
        query_data,
        model,
        index_mapping,
        output_path,
        index,
        max_seq_length,
        write_results,
    ) = args

    query_names, queries, _ = _calc_embeddings(query_data, model, max_seq_length)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("Searching...")
    for i in tqdm.tqdm(range(len(queries))):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        # filtered_scores = filter_scores(scores, reduce_indices(indices, index_mapping))
        filtered_scores = filter_scores(scores, indices)
        if write_results:
            f = open(f"{output_path}/{query_names[i]}.txt", "w")
            f.write("Name     Distance" + "\n")
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()


def est_nprobe(index, threshold=1):
    # Retrieve centroid vectors for each Voronoi cell
    centroids = index.quantizer.reconstruct_n(0, index.quantizer.ntotal).cpu()

    random_numbers = np.random.randint(1000, size=(10,))
    random_centroids = [centroids[i] for i in random_numbers]

    surrounding_cells = []
    for target_centroid in random_centroids:
        distances = np.linalg.norm(centroids - target_centroid, axis=1)
        sorted_distances_indices = np.argsort(distances)

        sorted_distances = distances[sorted_distances_indices]
        nprobe_est = len(np.where(sorted_distances < threshold)[0])
        surrounding_cells.append(nprobe_est)
    print(f"Estimating nprobe from {surrounding_cells}")
    nprobe_avg = np.mean(surrounding_cells)
    print(f"Nprobe: {nprobe_avg}")

    return int(nprobe_avg)
    # Calculate Voronoi cell boundaries (e.g., using Convex Hull algorithm)

    # Calculate angles between Voronoi cell boundaries


def evaluate(
    params, query_names, query_embeddings, index, unrolled_names, write_results
) -> dict:
    """Evaluation pipeline.

    Calculates embeddings for query and targets
    If visactmaps is true, generates activation map plots given the target embeddings
        (probably want to remove this feature)
    Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
    """

    filter(query_embeddings, query_names, params, index, unrolled_names, write_results)
