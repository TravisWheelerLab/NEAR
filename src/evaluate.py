import torch
import os
from types import SimpleNamespace
import time
import yaml
import argparse
import itertools
import tqdm
import h5py
import faiss
from src.evaluator.contrastive_evaluator import (
    _setup_targets_for_search,
    _calc_embeddings,
    search_and_filter,
)
from src.utils.eval_utils import split, load_model
from src.utils.faiss_utils import save_FAISS_results, load_index

from multiprocessing.pool import ThreadPool as Pool
import concurrent.futures
from src.utils.loaders import (
    load_model_class,
)
import pickle
import numpy as np
from src.data.hmmerhits import FastaFile
from src.utils import encode_string_sequence

HOME = os.environ["HOME"]


def batch_search(args):
    (i, sequences, model, output_path, index, max_seq_length) = args
    queries, _, _ = _calc_embeddings(sequences, model)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    lengths = [len(q) for q in queries]
    flattened_qs = torch.cat(queries, dim=0)

    search_time = time.time()
    scores, indices = index.search(flattened_qs.contiguous(), k=1000)
    search_time = time.time() - search_time
    all_scores = np.split(scores.to("cpu").numpy(), np.cumsum(lengths)[:-1])
    all_indices = np.split(indices.to("cpu").numpy(), np.cumsum(lengths)[:-1])
    print(f"Thread {i} completed search")

    return i, all_scores, all_indices, search_time


def search(args):
    (idx, sequences, model, output_path, index, max_seq_length, device) = args
    queries, _, _ = _calc_embeddings(sequences, model, device)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    all_scores = []
    all_indices = []
    search_time = time.time()
    for i in tqdm.tqdm(range(len(queries))):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        all_scores.append(scores)
        all_indices.append(indices)
    search_time = time.time() - search_time
    print(f"Thread {idx} completed search")
    return idx, all_scores, all_indices, search_time


def search_iter(args):
    (i, sequences, model, output_path, index, max_seq_length) = args
    queries, _, _ = _calc_embeddings(sequences, model)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    all_scores = []
    all_indices = []

    batches = list(split(queries, len(queries) // 100))
    search_time = time.time()

    for batch in tqdm.tqdm(batches):
        lengths = [len(q) for q in batch]
        flatq = torch.cat(batch, dim=0)
        scores, indices = index.search(flatq.contiguous(), k=1000)
        scores = np.split(scores.to("cpu").numpy(), np.cumsum(lengths)[:-1])
        indices = np.split(indices.to("cpu").numpy(), np.cumsum(lengths)[:-1])
        all_scores += scores
        all_indices += indices
    search_time = time.time() - search_time

    return i, all_scores, all_indices, search_time


def evaluate_multiprocessing(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")

    model_class = load_model_class(params.model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data
    print(f"Number of queries: {len(query_sequences)}")
    # q_chunk_size = len(query_sequences) // params.num_threads

    numqueries = len(query_sequences)
    if os.path.exists(params.index_path):
        index = faiss.read_index(params.index_path)
        index.nprobe = params.nprobe
    else:
        index = load_index(params, model, device)
    print(f"nprobe : {params.nprobe}")
    print(f"omp num threads: {params.omp_num_threads}")
    # faiss.omp_set_num_threads(params.omp_num_threads)
    split_queries = list(split(list(query_sequences.values()), params.num_threads))
    split_names = list(split(list(query_sequences.keys()), params.num_threads))
    print(len(split_queries))
    arg_list = [
        (
            # dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            i,
            split_queries[i],
            model,
            params.save_dir,
            index,
            params.max_seq_length,
        )
        for i in range(params.num_threads)
    ]
    del query_sequences
    print(f"Length of arg list: {len(arg_list)}")

    query_names_list = []
    all_scores_list = []
    all_indices_list = []
    print("Beginning search...")
    full_search_time = 0
    _concurrent = True
    start = time.time()

    if _concurrent:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=params.num_threads
        ) as executor:
            future_to_batch = {
                executor.submit(search, batch): batch for batch in arg_list
            }

        # Collect results as they become available
        for future in concurrent.futures.as_completed(future_to_batch):
            #        batch = future_to_batch[future]
            i, all_scores, all_indices, search_time = future.result()
            query_names_list += split_names[i]
            all_scores_list += all_scores
            all_indices_list += all_indices  # ... combine results ...
            full_search_time += search_time
    else:
        pool = Pool(params.num_threads)
        for result in pool.imap(search, arg_list):
            i, all_scores, all_indices, search_time = result
            query_names_list += split_names[i]
            all_scores_list += all_scores
            all_indices_list += all_indices  # ... combine results ...
            full_search_time += search_time

    assert len(all_scores_list) == numqueries
    print(
        f"Search time per query: {(full_search_time)/(params.num_threads*numqueries)}."
    )
    print(f"Elapsed time per query: {(time.time() - start)/numqueries}.")
    if params.write_results:
        save_FAISS_results(
            query_names_list,
            all_scores_list,
            all_indices_list,
            params.scores_path,
            params.indices_path,
            params.query_names_path,
        )


def evaluate(_config):
    params = SimpleNamespace(**_config)
    print(params.num_threads)
    print(f"Loading from checkpoint in {params.checkpoint_path}")

    model_class = load_model_class(params.model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data
    if os.path.exists(params.index_path):
        index = faiss.read_index(params.index_path)
        index.nprobe = params.nprobe
        if params.device == "cuda":
            num = 0
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, int(num), index)
    else:
        index = load_index(params)
    # faiss.omp_set_num_threads(params.omp_num_threads)
    # index = load_index(params)
    query_names = list(query_sequences.keys())
    arg_list = [
        0,
        list(query_sequences.values()),
        model,
        #        index_mapping,
        params.save_dir,
        index,
        params.max_seq_length,
        params.device,
    ]
    # del query_sequences
    print(f"Number of queries: {len(query_sequences)}")
    numqueries = len(query_sequences)
    del query_sequences
    print("Beginning search...")
    start = time.time()

    _, all_scores, all_indices, search_time = search(arg_list)

    print(f"Search time per query: {(search_time)/(params.num_threads*numqueries)}.")
    print(f"Elapsed time per query: {(time.time() - start)/numqueries}.")

    if params.write_results:
        save_FAISS_results(
            query_names,
            all_scores,
            all_indices,
            params.scores_path,
            params.indices_path,
            params.query_names_path,
        )


def evaluate_python(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")

    model_class = load_model_class(params.model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    index, index_mapping = load_index(params)

    arg_list = [
        query_sequences,
        model,
        index_mapping,
        params.save_dir,
        index,
        params.max_seq_length,
        params.write_results,
    ]
    del query_sequences

    print("Beginning search...")
    start = time.time()

    search_and_filter(arg_list)

    print(f"Elapsed time: {time.time() - start}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--rust", action="store_true")

    args = parser.parse_args()
    configfile = args.config.strip(".yaml")

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)
    if _config["num_threads"] > 1:
        if args.rust:
            print("Rust evaluation pipeline")
            evaluate_multiprocessing(_config)
        else:
            print("Python evaluation pipeline")
            evaluate_multiprocessing_python(_config)
    else:
        if args.rust:
            print("Rust evaluation pipeline")
            evaluate(_config)
        else:
            print("Python evaluation pipeline")
            evaluate_python(_config)
