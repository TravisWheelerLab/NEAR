"""
Final version of the evaluator class. 
This is preferred when using multiprocessing """

import logging
import os
import tqdm
import numpy as np
import torch
import time
from src.utils import encode_string_sequence
from collections import defaultdict
import pdb

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
        if len(sequence) > 0:
            embed = (
                model_class(
                    encode_string_sequence(sequence).unsqueeze(0).to(model_device)
                )
                .squeeze()
                .T
            )
            embeddings.append(torch.nn.functional.normalize(embed, dim=-1).to("cpu"))
            lengths.append(len(sequence))
            indices.append(idx)
    return embeddings, lengths, indices


@torch.no_grad()
def search(args):
    (idx, sequences, model, output_path, index, unrolled_lengths, device) = args
    queries, _, query_indices = _calc_embeddings(sequences, model, device)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    all_scores = []
    all_indices = []
    unrolled_lengths = np.array(unrolled_lengths).astype(int)
    search_time = time.time()

    for i in tqdm.tqdm(range(len(queries))):
        # searching all amino acids in queries[i]
        # returns a list of 1000 scores per amino and the indices of the target sequences per amino
        scores, indices = index.search(queries[i].contiguous().numpy(), k=1000)
        norm_factors = np.array(
            [len(queries[i]) * unrolled_lengths[ind] for ind in indices]
        )  # this should be an array of shape (len(queries[i]), 1000))
        normalized_scores = scores / norm_factors

        all_scores.append(normalized_scores)
        all_indices.append(indices)
    search_time = time.time() - search_time
    print(f"Thread {idx} completed search")
    return idx, all_scores, all_indices, search_time, query_indices


@torch.no_grad()
def search_and_filter(args):
    (
        query_data,
        model,
        output_path,
        index,
        unrolled_lengths,
        write_results,
    ) = args

    query_names = np.array(list(query_data.keys()))

    queries, _, indices = _calc_embeddings(list(query_names.values()), model)

    query_names = query_names[indices]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print("Searching...")
    for i in tqdm.tqdm(range(len(queries))):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        normalized_scores = [
            score / (len(queries[i]) * len(unrolled_lengths[ind]))
            for score, ind in zip(scores, indices)
        ]
        filtered_scores = filter_scores(normalized_scores, indices)
        if write_results:
            f = open(f"{output_path}/{query_names[i]}.txt", "w")
            f.write("Name     Distance" + "\n")
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()
