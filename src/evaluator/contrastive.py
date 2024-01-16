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


def filter_scores(
    scores_array,
    indices_array,
    unrolled_names,
    normalise_by_target=False,
):
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


def filter_sequences_by_mask(masked_targets, embeddings):
    filtered_embeddings = []
    filtered_lengths = []

    masked_target_sequences = [m for m in masked_targets if len(m) > 0]

    assert len(masked_target_sequences) == len(embeddings)

    for sequence, embedding in tqdm.tqdm(zip(masked_target_sequences, embeddings)):
        assert len(sequence) == len(embedding)
        Xs = [i for i in range(len(sequence)) if sequence[i] == "X"]
        embedding = np.delete(embedding, Xs, axis=0)
        filtered_embeddings.append(embedding)
        filtered_lengths.append(len(embedding))

    return filtered_embeddings, filtered_lengths


@torch.no_grad()
def _calc_embeddings(
    sequences,
    model_class,
    model_device="cpu",
    max_seq_length=512,
):
    sequences, model_class, device = args
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
    (
        idx,
        query_sequences,
        masked_query_sequences,
        model,
        index,
        device,
        mask_queries,
    ) = args
    queries, _, query_indices = _calc_embeddings(query_sequences, model, device)
    if mask_queries:
        queries, _ = filter_sequences_by_mask(masked_query_sequences, queries)
    all_scores = []
    all_indices = []
    search_time = time.time()

    for i in tqdm.tqdm(range(len(queries))):
        # searching all amino acids in queries[i]
        # returns a list of 1000 scores per amino and the indices of the target sequences per amino
        scores, indices = index.search(queries[i].contiguous().numpy(), k=1000)

        all_scores.append(scores)
        all_indices.append(indices)
    # pdb.set_trace()
    search_time = time.time() - search_time
    print(f"Thread {idx} completed search")
    return idx, all_scores, all_indices, search_time, query_indices


@torch.no_grad()
def search_and_filter(args):
    (
        query_data,
        masked_query_data,
        model,
        output_path,
        index,
        write_results,
        unrolled_names,
        mask_queries,
        device,
    ) = args

    query_names = np.array(list(query_data.keys()))

    queries, _, indices = _calc_embeddings(list(query_data.values()), model, device)

    if mask_queries:
        queries, _ = filter_sequences_by_mask(list(masked_query_data.values()), queries)

    query_names = query_names[indices]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    unrolled_names = np.array(unrolled_names)
    print("Searching...")

    for i in tqdm.tqdm(range(len(queries))):
        scores, indices = index.search(queries[i].contiguous().numpy(), k=1000)

        filtered_scores = filter_scores(scores, indices, unrolled_names)

        if write_results:
            f = open(f"{output_path}/{query_names[i]}.txt", "w")
            f.write("Name     Distance" + "\n")
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()
