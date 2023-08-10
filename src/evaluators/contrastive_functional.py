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
import my_rust_module
import sys
from collections import defaultdict
import pdb
logger = logging.getLogger("evaluate")


def filter_scores(
    scores_array: np.array, indices_array: np.array, unrolled_names: np.array
) -> dict:
    """Filters the scores such that every query amino can only
    be matched to one amino from each target sequence
    and it matches the one with the biggest score.

    Then sums the scores that belong to the same target
    and returns the resulting distances in a dict

    scores_array (numqueryaminos, 1000): an array of 1000 scores per query amino
    indices_array: (numqueryaminos, 1000) the indices of the target sequence name (in unrolled_names)
    for each of the scores in scores_array
    unrolled_names: an array of target names that the indices in indices_array correspond to
    """

    #filtered_scores_list = []
    #for scores_array, indices_array in zip(scores_array_list, indices_array_list):
    filtered_scores: dict = defaultdict(float)

        # iterate over query amino scores
    for match_idx in range(len(scores_array)):
        match_scores = scores_array[match_idx]
        names = unrolled_names[
            indices_array[match_idx]
        ]  # the names of the targets for each 1000 hits
        sorted_match_idx = np.argsort(match_scores)[::-1]

        #pdb.set_trace()           
        _, unique_indices = np.unique(names[sorted_match_idx], return_index=True)
        new_indices = list(
            indices_array[match_idx][sorted_match_idx][unique_indices]
        )
        new_scores = list(match_scores[sorted_match_idx][unique_indices])

        for distance, name in zip(new_scores, unrolled_names[new_indices]):
            filtered_scores[name] += distance
        #filtered_scores_list.append(filtered_scores)

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


def filter_and_calc_embeddings2(
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
    #    print(sequences[0])
    # filtered_names = names.copy()
    num_removed = 0
    # print("Calculating embeddings...")
    for sequence in sequences:
        length = len(sequence)
        #        print(length)
        if max_seq_length >= length >= minimum_seq_length:
            embed = (
                model_class(
                    encode_string_sequence(sequence).unsqueeze(0).to(model_device)
                )
                .squeeze()
                .T
            )
            #            print(embed.shape)
            # return: seq_lenxembed_dim shape
            embeddings.append(torch.nn.functional.normalize(embed, dim=-1).to("cpu"))
            lengths.append(length)
        else:
            num_removed += 1
            # filtered_names.remove(name)
            # filtered_sequences.remove(sequence)
    return embeddings, lengths


@torch.no_grad()
def _calc_embeddings(
    sequence_data, model_class, max_seq_length
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """Calculates the embeddings for the sequences by
    calling the model forward function. Filters the sequences by max/min
    sequence length and returns the filtered sequences/names and embeddings

    Returns [names], [sequences], [embeddings]"""

    names = list(sequence_data.keys())
    sequences = list(sequence_data.values())

    filtered_names, embeddings, lengths = filter_and_calc_embeddings(
        names, sequences, model_class, max_seq_length
    )

    return filtered_names, embeddings, lengths


def search(
    index, index_mapping, target_names, query_embedding: torch.Tensor
) -> List[Tuple[str, float]]:
    """Searches through the target DB and gathers a
    filtered list of sequences and distances to their centre
    which we use as hits for the given query"""

    search_start = time.time()

    scores_array, indices_array = index.search(query_embedding.contiguous(), k=1000)

    indices_array = reduce_indices(indices_array.to("cpu").numpy(), index_mapping)

    search_time = time.time() - search_start

    filtration_time = time.time()

    filtered_scores = filter_scores(
        scores_array.to("cpu").numpy(), indices_array, target_names
    )

    filtration_time = time.time() - filtration_time

    # TODO: divide by query sequence length and multiply by median sequence length
    return filtered_scores, search_time, filtration_time


def save_target_embeddings(arg_list):
    target_data, model, max_seq_length = arg_list

    targets, lengths = _calc_embeddings(target_data, model, max_seq_length)

    return targets, lengths


def reduce_indices(indices, index_mapping):
    # indices is of shape (seq len, 1000)
    new_indices = np.zeros_like(indices)

    for i, amino_index_list in enumerate(indices):
        # new_indices[i] = np.array(
        #    [names.index(unrolled_names[idx]) for idx in amino_index_list]
        # )
        new_indices[i] = np.array([index_mapping[idx] for idx in amino_index_list])

    return new_indices


def search_only(args):
    (query_data, model, output_path, index, max_seq_length) = args
    query_names, queries, _ = _calc_embeddings(query_data, model, max_seq_length)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #    start_time = time.time()

    all_scores = []
    all_indices = []

    print("Searching...")
    for i in range(len(queries)):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        all_scores.append(scores.to("cpu").numpy())

        all_indices.append(indices.to("cpu").numpy())

    #    total_search_time = time.time() - start_time

    return query_names, all_scores, all_indices


def filter_only(arg_list):
    (
        all_scores,
        all_indices,
        unrolled_names,
        query_names,
        write_results,
        output_path,
    ) = arg_list
    filtration_time = time.time()
    # Call the filter_scores function from the Rust module
    # print("Calling rust function...")
    filtered_scores_list = my_rust_module.filter_scores(
        all_scores, all_indices, unrolled_names
    )

    # print("Filtration complete")

    total_filtration_time = time.time() - filtration_time

    assert len(filtered_scores_list) == len(query_names)

    if write_results:
        for i, filtered_scores in enumerate(filtered_scores_list):
            f = open(f"{output_path}/{query_names[i]}.txt", "w")
            f.write("Name     Distance" + "\n")
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()
    return total_filtration_time


@torch.no_grad()
def filter(arg_list):
    """Filters our hits based on
    distance to the query in the Faiiss
    cluster space"""

    (
        query_data,
        model,
        output_path,
        index,
        target_names,
        index_mapping,
        max_seq_length,
        write_results,
    ) = arg_list

    query_names, queries, _ = _calc_embeddings(query_data, model, max_seq_length)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    start_time = time.time()

    total_search_time = 0
    total_filtration_time = 0

    for i in tqdm.tqdm(range(len(queries))):
        filtered_scores, search_time, filtration_time = search(
            index, index_mapping, target_names, queries[i]
        )
        total_search_time += search_time

        total_filtration_time += filtration_time

        if write_results:
            f = open(f"{output_path}/{query_names[i]}.txt", "w")
            f.write("Name     Distance" + "\n")
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()
    duration = time.time() - start_time
    return duration, total_search_time, total_filtration_time


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


def _setup_targets_for_search(
    target_embeddings: List[torch.Tensor],
    target_names: List[str],
    lengths: List[int],
    index_string,
    nprobe,
    num_threads=1,
    normalize_embeddings=True,
    index_device="cpu",
    index_path="/xdisk/twheeler/daphnedemekas/faiss-index-targets.index",
):
    """Creates the Faiss Index object using the unrolled
    target embddings"""
    faiss.omp_set_num_threads(num_threads)

    unrolled_names = np.repeat(target_names, lengths)
    unrolled_targets = torch.cat(
        target_embeddings, dim=0
    )  # (num targets x amino per target) x 256

    del lengths

    unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

    start = time.time()
    if not os.path.exists(index_path):
        print(f"Creating index: {index_string} and saving to {index_path}")

        index: faiss.Index = create_faiss_index(
            embeddings=unrolled_targets,
            embed_dim=unrolled_targets.shape[-1],
            distance_metric="cosine" if normalize_embeddings else "l2",
            index_string=index_string,  # f"IVF{K},PQ8", #self.index_string, #f"IVF100,PQ8", #"IndexIVFFlat", #self.index_string,
            device=index_device,
        )
        logger.info("Adding targets to index.")
        if index_device == "cpu":
            index.add(unrolled_targets.to("cpu"))
        else:
            index.add(unrolled_targets)
        faiss.write_index(index, index_path)
    else:
        print(f"Reading index from {index_path}")
        index = faiss.read_index(index_path)

    index.nprobe = nprobe
    loop_time = time.time() - start

    print(f"Index Creation took: {loop_time}.")

    return unrolled_names, index


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

