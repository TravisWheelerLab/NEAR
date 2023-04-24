
"""" Evaluator class for the contrastive CNN model """

import logging
import os
from typing import List, Tuple
import tqdm 
import faiss
import numpy as np
import torch

from src.utils import create_faiss_index, encode_string_sequence
import time
logger = logging.getLogger("evaluate")


def filter_scores(scores_array, indices_array, unrolled_names):
    """Filters the scores such that every query amino can only
    be matched to one amino from each target sequence
    and it matches the one with the biggest score"""
    scores = []
    indices = []

    for idx in range(len(scores_array)):
        scores_idx = scores_array[idx]
        names = unrolled_names[
            indices_array[idx]
        ]  # the names of the targets for each 1000 hits
        sorted_idx = np.argsort(scores_idx)[::-1]

        _, unique_indices = np.unique(
            names[sorted_idx], return_index=True
        )  # the unique names of the targets for each 1000 hits (<= 1000)
        indices += list(indices_array[idx][sorted_idx][unique_indices])
        scores += list(scores_idx[sorted_idx][unique_indices])

    return scores, indices


def filter_sequences_by_length(
        names: List[str],
        sequences: List[str],
        model_class,
        model_device = 'cpu',
        max_seq_length = 512, 
        minimum_seq_length = 0,
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Filters the sequences by length thresholding given the
        minimum and maximum length threshold variables"""
        embeddings = []
        lengths = []

        filtered_names = names.copy()
        num_removed = 0
        for name, sequence in zip(names, sequences):
            length = len(sequence)
            if max_seq_length >= length >= minimum_seq_length:
                embed = (model_class(encode_string_sequence(sequence).unsqueeze(0).to(model_device))
                        .squeeze()
                        .T)
                # return: seq_lenxembed_dim shape
                embeddings.append(embed.to("cpu"))
                lengths.append(length)
            else:
                num_removed += 1
                filtered_names.remove(name)
                # filtered_sequences.remove(sequence)
        return filtered_names, embeddings, lengths

@torch.no_grad()
def _calc_embeddings(sequence_data, model_class) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """Calculates the embeddings for the sequences by
    calling the model forward function. Filters the sequences by max/min
    sequence length and returns the filtered sequences/names and embeddings

    Returns [names], [sequences], [embeddings]"""
    
    names = list(sequence_data.keys())
    sequences = list(sequence_data.values())

    filtered_names, embeddings, lengths = filter_sequences_by_length(
        names, sequences, model_class
    )

    return filtered_names, embeddings, lengths

def search(index, unrolled_names, query_embedding: torch.Tensor) -> List[Tuple[str, float]]:
        """Searches through the target DB and gathers a
        filtered list of sequences and distances to their centre
        which we use as hits for the given query"""

        filtered_scores = {}
        scores_array, indices_array = index.search(query_embedding.contiguous(), k=1000)

        scores, indices = filter_scores(
            scores_array.to("cpu").numpy(), indices_array.to("cpu").numpy(), unrolled_names
        )

        for distance, name in zip(scores, unrolled_names[indices],):
            # filtered_list.append((name, distance))
            if name in filtered_scores.keys():
                filtered_scores[name] += distance
            else:
                filtered_scores[name] = distance

        return filtered_scores


@torch.no_grad()
def filter(arg_list, normalize_embeddings = True
):
    """Filters our hits based on
    distance to the query in the Faiiss
    cluster space"""
    
    query_data, model, output_path, index, unrolled_names = arg_list
    
    query_names, queries, _ = _calc_embeddings(query_data, model)

    #logger.info("Beginning search.")

    #t_begin = time.time()

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #print(f"Number of queries: {len(queries)}")
    # all_filtered_scores = {}
    # for i in tqdm.tqdm(range(len(queries))):

    #     if normalize_embeddings:
    #         qval = torch.nn.functional.normalize(queries[i], dim=-1)
    #     else:
    #         qval = queries[i]

    #     filtered_scores = search(index, unrolled_names, qval)
    #     all_filtered_scores[i] = filtered_scores
    for i in tqdm.tqdm(range(len(queries))):

        f = open(f"{output_path}/{query_names[i]}.txt", "w")
        f.write("Name     Distance" + "\n")

        qval = torch.nn.functional.normalize(queries[i], dim=-1)

        filtered_scores = search(index, unrolled_names, qval)#, search_time, filter_time, aggregate_time)
        #all_filtered_scores[i] = filtered_scores
        for name, distance in filtered_scores.items():
            f.write(f"{name}     {distance}" + "\n")
        f.close()

    #loop_time = time.time() - t_begin

    #logger.info(f"Entire loop took: {loop_time}.")

    #print("Writing results to file...")
    # for query_idx, score in all_filtered_scores.items():
    #     f = open(f"{output_path}/{query_names[query_idx]}.txt", "w")
    #     f.write("Name     Distance" + "\n")
    #     for name, distance in score.items():
    #         f.write(f"{name}     {distance}" + "\n")
    #     f.close()

def _setup_targets_for_search(
        target_embeddings: List[torch.Tensor], target_names: List[str], lengths: List[int], index_string, nprobe, normalize_embeddings=True, index_device = "cpu"):
    """Creates the Faiss Index object using the unrolled
    target embddings"""

    unrolled_names = np.repeat(target_names, lengths)
    unrolled_targets = torch.cat(
        target_embeddings, dim=0
    )  # (num targets x amino per target) x 256

    del lengths

    unrolled_targets = torch.nn.functional.normalize(unrolled_targets, dim=-1)

    index: faiss.Index = create_faiss_index(
        embeddings=unrolled_targets,
        embed_dim=unrolled_targets.shape[-1],
        distance_metric="cosine" if normalize_embeddings else "l2",
        index_string=index_string,  # f"IVF{K},PQ8", #self.index_string, #f"IVF100,PQ8", #"IndexIVFFlat", #self.index_string,
        nprobe=nprobe,
        device=index_device,
    )

    logger.info("Adding targets to index.")
    if index_device == "cpu":
        index.add(unrolled_targets.to("cpu"))
    else:
        index.add(unrolled_targets)

    faiss.omp_set_num_threads(int(os.environ.get("NUM_THREADS")))
    
    return unrolled_names, index


def evaluate(params, query_names, query_embeddings, index, unrolled_names) -> dict:
    """Evaluation pipeline.

    Calculates embeddings for query and targets
    If visactmaps is true, generates activation map plots given the target embeddings
        (probably want to remove this feature)
    Then runs Faiss clustering and filtering and returns a dictionary of the model's hits.
    """

    filter(query_embeddings, query_names, params, index, unrolled_names)
