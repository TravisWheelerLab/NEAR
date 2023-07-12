import torch
import torch.profiler as profiler
from typing import List, Tuple
import tqdm
from src.utils import create_faiss_index, encode_string_sequence
import itertools
from multiprocessing.pool import ThreadPool as Pool

import time


def embed_with_profile(
    arg_list,
    max_seq_length=512,
    model_device="cpu",
    minimum_seq_length=0,
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """Filters the sequences by length thresholding given the
    minimum and maximum length threshold variables"""

    names, sequences, model_class = arg_list
    embeddings = []
    lengths = []

    filtered_names = names.copy()
    num_removed = 0
    for name, sequence in tqdm.tqdm(zip(names, sequences)):
        length = len(sequence)
        if max_seq_length >= length >= minimum_seq_length:
            # with profiler.profile(profile_memory = True, record_shapes = True, use_cuda = True) as prof:
            embed = (
                model_class(encode_string_sequence(sequence).unsqueeze(0).to(model_device))
                .squeeze()
                .T
            )
            # return: seq_lenxembed_dim shape
            embeddings.append(embed.to("cpu"))
            lengths.append(length)
            # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=3))
        else:
            num_removed += 1
            filtered_names.remove(name)
            # filtered_sequences.remove(sequence)
    return filtered_names, embeddings, lengths


def embed_multithread(query_sequences, model, q_chunk_size, num_threads=16):

    arg_list = [
        (list(data.keys()), list(data.values()), model)
        for data in [
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size))
            for i in range(0, len(query_sequences), q_chunk_size)
        ]
    ]

    pool = Pool(num_threads)

    print("Beginning search...")

    start_time = time.time()

    for result in pool.imap(embed_with_profile, arg_list):
        print("Got result")

    loop_time = time.time() - start_time
    print(f"Entire search took: {loop_time}.")
    pool.terminate()


@torch.no_grad()
def profile_embeddings(
    sequence_data, model_class, max_seq_length
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """Calculates the embeddings for the sequences by
    calling the model forward function. Filters the sequences by max/min
    sequence length and returns the filtered sequences/names and embeddings

    Returns [names], [sequences], [embeddings]"""

    names = list(sequence_data.keys())
    sequences = list(sequence_data.values())

    t_begin = time.time()

    filtered_names, embeddings, lengths = embed_with_profile(
        [names, sequences, model_class], max_seq_length
    )
    loop_time = time.time() - t_begin

    print(f"Entire loop took: {loop_time}.")
    print(len(filtered_names))
    return filtered_names, embeddings, lengths
