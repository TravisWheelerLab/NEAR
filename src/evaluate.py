import os
from types import SimpleNamespace
import time
import yaml
import argparse
import itertools
from src.evaluator.contrastive import (
    search,
    search_and_filter,
)
from src.utils.eval_utils import split, load_model
from src.utils.faiss_utils import save_FAISS_results, load_index

from multiprocessing.pool import ThreadPool as Pool
import concurrent.futures
import numpy as np
from src.data.hmmerhits import FastaFile

HOME = os.environ["HOME"]


# def evaluate_multiprocessing(_config):
#     """Search component of the evaluation pipeline with multiprocessing
#     Results are saved to .h5 files to be post-processed in rust."""
#     params = SimpleNamespace(**_config)

#     print(f"Loading from checkpoint in {params.checkpoint_path}")

#     model = load_model(params.checkpoint_path, params.model_name, params.device)

#     queryfasta = FastaFile(params.query_file)
#     query_sequences = queryfasta.data
#     print(f"Number of queries: {len(query_sequences)}")

#     numqueries = len(query_sequences)
#     index = load_index(params, model)
#     print(f"nprobe : {params.nprobe}")
#     print(f"omp num threads: {params.omp_num_threads}")
#     # faiss.omp_set_num_threads(params.omp_num_threads)
#     split_queries = list(split(list(query_sequences.values()), params.num_threads))
#     split_names = list(split(list(query_sequences.keys()), params.num_threads))

#     arg_list = [
#         (
#             i,
#             split_queries[i],
#             model,
#             params.save_dir,
#             index,
#             params.device,
#         )
#         for i in range(params.num_threads)
#     ]
#     del query_sequences

#     query_names_list = []
#     all_scores_list = []
#     all_indices_list = []
#     print("Beginning search...")
#     full_search_time = 0
#     start = time.time()

#     with concurrent.futures.ThreadPoolExecutor(
#         max_workers=params.num_threads
#     ) as executor:
#         future_to_batch = {executor.submit(search, batch): batch for batch in arg_list}

#     # Collect results as they become available
#     for future in concurrent.futures.as_completed(future_to_batch):
#         #        batch = future_to_batch[future]
#         i, all_scores, all_indices, search_time, query_indices = future.result()
#         query_names_list += list(np.array(split_names[i])[query_indices])
#         all_scores_list += list(all_scores)
#         all_indices_list += list(all_indices)  # ... combine results ...
#         full_search_time += search_time

#     if not len(all_scores_list) == numqueries:
#         print("Warning: not all queries are returned")
#         print(f"num scores: {len(all_scores_list)}, num queries : {numqueries}")
#     print(
#         f"Search time per query: {(full_search_time)/(params.num_threads*numqueries)}."
#     )
#     print(f"Elapsed time per query: {(time.time() - start)/numqueries}.")
#     if params.write_results:
#         save_FAISS_results(
#             query_names_list,
#             all_scores_list,
#             all_indices_list,
#             params.scores_path,
#             params.indices_path,
#             params.query_names_path,
#         )


def evaluate_multiprocessing_python(_config):
    """Full evaluation pipeline running parallelized with the
    post-processing in python."""
    params = SimpleNamespace(**_config)

    model = load_model(params.checkpoint_path, params.model_name, params.device)

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")

    index, unrolled_names = load_index(params, model, params.mask_targets)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data
    maskedqueryfasta = FastaFile(params.masked_query_file)
    masked_queries = maskedqueryfasta.data

    q_chunk_size = len(query_sequences) // params.num_threads

    numqueries = len(query_sequences)
    print(f"Number of queries: {numqueries}")
    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            dict(itertools.islice(masked_queries.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            index,
            params.write_results,
            unrolled_names,
            params.mask_queries,
            params.normalise_search_results,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences
    pool = Pool(params.num_threads)

    print("Beginning search...")
    start = time.time()

    idx = 0
    for result in pool.imap(search_and_filter, arg_list):
        print(f"Finished thread: {idx}")

    print(f"Elapsed time: {time.time() - start}.")

    pool.terminate()


# def evaluate(_config):
#     """Search component of the evaluation pipeline without multiprocessing
#     Results are saved to .h5 files to be post-processed in rust."""
#     params = SimpleNamespace(**_config)
#     print(f"Loading from checkpoint in {params.checkpoint_path}")

#     model = load_model(params.checkpoint_path, params.model_name, params.device)

#     queryfasta = FastaFile(params.query_file)
#     query_sequences = queryfasta.data
#     index = load_index(params, model)
#     # index = load_index(params)
#     with open(params.unrolled_lengths, "r") as f:
#         unrolled_lengths = [int(line.strip()) for line in f.readlines()]
#     query_names = list(query_sequences.keys())
#     arg_list = [
#         0,
#         list(query_sequences.values()),
#         model,
#         params.save_dir,
#         index,
#         unrolled_lengths,
#         params.device,
#     ]
#     print(f"Number of queries: {len(query_sequences)}")
#     numqueries = len(query_sequences)
#     del query_sequences

#     print("Beginning search...")
#     start = time.time()

#     _, all_scores, all_indices, search_time, query_indices = search(arg_list)

#     query_names = np.array(query_names)[query_indices]

#     print(f"Search time per query: {(search_time)/(params.num_threads*numqueries)}.")
#     print(f"Elapsed time per query: {(time.time() - start)/numqueries}.")

#     if params.write_results:
#         save_FAISS_results(
#             query_names,
#             all_scores,
#             all_indices,
#             params.scores_path,
#             params.indices_path,
#             params.query_names_path,
#         )


# def evaluate_python(_config):
#     """Full evaluation pipeline running unparallelized with the
#     post-processing in python."""
#     params = SimpleNamespace(**_config)

#     model = load_model(params.checkpoint_path, params.model_name, params.device)

#     queryfasta = FastaFile(params.query_file)
#     query_sequences = queryfasta.data

#     index = load_index(params, model)
#     with open(params.unrolled_lengths, "r") as f:
#         unrolled_lengths = [int(line.strip()) for line in f.readlines()]
#     arg_list = [
#         query_sequences,
#         model,
#         params.save_dir,
#         index,
#         unrolled_lengths,
#         params.write_results,
#     ]
#     del query_sequences

#     print("Beginning search...")
#     start = time.time()

#     search_and_filter(arg_list)

#     print(f"Elapsed time: {time.time() - start}.")


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
            print("FAISS scores saved. Run post-processing in rust for results ")
        else:
            print("Python evaluation pipeline")
            evaluate_multiprocessing_python(_config)
    else:
        if args.rust:
            print("Rust evaluation pipeline")
            evaluate(_config)
            print("FAISS scores saved. Run post-processing in rust for results ")
        else:
            print("Python evaluation pipeline")
            evaluate_python(_config)
