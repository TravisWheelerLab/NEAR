import torch
import os
from types import SimpleNamespace
import time
import yaml
import argparse
import itertools
from src.evaluators.contrastive_functional import (
    filter,
    _setup_targets_for_search,
    save_target_embeddings,
    search_only,
    filter_only,
)
from src.evaluators.profiler import profile_embeddings, embed_multithread
from src.data.hmmerhits import FastaFile
from src.data.eval_utils import get_evaluation_data

from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool as MPool
from src.utils.util import (
    load_dataset_class,
    load_evaluator_class,
    load_model_class,
)
import pickle
import pdb
import my_rust_module

HOME = os.environ["HOME"]


def save_off_targets(
    target_sequences, num_threads, model, max_seq_length, device, savedir
):
    t_chunk_size = len(target_sequences) // num_threads

    arg_list = [
        (
            dict(itertools.islice(target_sequences.items(), i, i + t_chunk_size)),
            model,
            max_seq_length,
        )
        for i in range(0, len(target_sequences), t_chunk_size)
    ]
    del target_sequences

    pool = Pool(num_threads)

    print("Embedding targets...")

    start_time = time.time()

    target_names = []
    target_embeddings = []
    target_lengths = []

    for result in pool.imap(save_target_embeddings, arg_list):
        names, embeddings, lengths = result
        target_names += names
        target_lengths += lengths
        target_embeddings += embeddings

    torch.save(target_embeddings, savedir)
    with open(f"{savedir.strip('.pt')}_names.pickle", "wb") as handle:
        pickle.dump(target_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{savedir.strip('.pt')}_lengths.pickle", "wb") as handle:
        pickle.dump(target_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loop_time = time.time() - start_time
    print(f"Embedding took: {loop_time}.")

    return target_names, target_lengths, target_embeddings


def load_targets(
    target_embeddings,
    target_names,
    target_lengths,
    target_file,
    num_threads,
    model,
    max_seq_length,
    device,
):
    # get target embeddings
    if not os.path.exists(target_embeddings):
        print("No saved target embeddings. Calculating them now.")
        targetfasta = FastaFile(target_file)
        target_sequences = targetfasta.data

        target_names, target_lengths, target_embeddings = save_off_targets(
            target_sequences,
            num_threads,
            model,
            max_seq_length,
            device,
            target_embeddings,
        )
    else:
        target_embeddings = torch.load(target_embeddings)

        if target_names.endswith(".pickle"):
            with open(target_names, "rb") as file_handle:
                target_names = pickle.load(file_handle)

            with open(target_lengths, "rb") as file_handle:
                target_lengths = pickle.load(file_handle)

        elif target_names.endswith(".txt"):
            with open(target_names, "r") as f:
                target_names = f.readlines()
                target_names = [t.strip("\n") for t in target_names]
            with open(target_lengths, "r") as f:
                target_lengths = f.readlines()
                target_lengths = [int(t.strip("\n")) for t in target_lengths]

        else:
            raise Exception("Saved target data format not understood")

    return target_embeddings, target_names, target_lengths


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def evaluate_for_times_mp(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")

    model_class = load_model_class(params.model_name)

    print(f"Index path: {params.index_path}")

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    query_sequences = {
        k: v
        for k, v in zip(
            list(query_sequences.keys())[:500], list(query_sequences.values())[:500]
        )
    }

    numqueries = len(query_sequences)
    print(f"Number of queries: {numqueries}")

    q_chunk_size = len(query_sequences) // params.num_threads

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")
    print(f"Chunk size: {q_chunk_size}")

    target_embeddings, target_names, target_lengths = load_targets(
        params.target_embeddings,
        params.target_names,
        params.target_lengths,
        params.target_file,
        params.num_threads,
        model,
        params.max_seq_length,
        params.device,
    )
    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
        index_path=params.index_path,
    )

    # index.parallel_mode = 1
    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            index,
            params.max_seq_length,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences

    pool = Pool(params.num_threads)
    print(f"Length of arglist: {len(arg_list)}")

    print("Searching with FAISS...")
    start = time.time()

    all_scores = []
    all_indices = []
    query_names_list = []
    for result in pool.imap(search_only, arg_list):
        search_time, query_names, scores, indices = result
        query_names_list += query_names
        all_scores += scores
        all_indices += indices
    search_time = time.time() - start

    pool.terminate()
    print(f"Search time: {search_time}")
    print(f"Search time per query: {search_time/(numqueries)}")

    print("Filtering in Rust...")

    total_filtration_time = time.time()
    filtered_scores_list = my_rust_module.filter_scores(
        all_scores, all_indices, unrolled_names
    )

    total_filtration_time = time.time() - total_filtration_time
    print(f"Filtration time: {total_filtration_time}.")
    print(f"Filtration time per query: {total_filtration_time/(numqueries)}.")

    elapsed_time = time.time() - start

    assert len(filtered_scores_list) == len(query_names_list)

    if params.write_results:
        for i, filtered_scores in enumerate(filtered_scores_list):
            f = open(f"{params.save_dir}/{query_names_list[i]}.txt", "w")
            f.write("Name     Distance" + "\n")
            for name, distance in filtered_scores.items():
                f.write(f"{name}     {distance}" + "\n")
            f.close()
    print(f"Elapsed time: {elapsed_time}")
    print(f"Elapsed time per query: {elapsed_time/(numqueries)}")


def evaluate_for_times_mp2(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")

    model_class = load_model_class(params.model_name)

    print(f"Index path: {params.index_path}")

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    #    query_sequences = {
    #        k: v
    #        for k, v in zip(
    #            list(query_sequences.keys())[:500], list(query_sequences.values())[:500]
    #        )
    #    }

    numqueries = len(query_sequences)
    print(f"Number of queries: {numqueries}")

    q_chunk_size = len(query_sequences) // params.num_threads

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")
    print(f"Chunk size: {q_chunk_size}")

    target_embeddings, target_names, target_lengths = load_targets(
        params.target_embeddings,
        params.target_names,
        params.target_lengths,
        params.target_file,
        params.num_threads,
        model,
        params.max_seq_length,
        params.device,
    )
    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
        index_path=params.index_path,
    )

    # index.parallel_mode = 1
    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            index,
            params.max_seq_length,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences

    pool = Pool(params.num_threads)
    print(f"Length of arglist: {len(arg_list)}")

    print("Searching with FAISS...")
    start = time.time()

    all_scores = []
    all_indices = []
    # query_names_list = []
    for result in pool.imap(search_only, arg_list):
        search_time, query_names, scores, indices = result
        #    query_names_list += query_names
        all_scores += scores
        all_indices += indices
    search_time = time.time() - start

    pool.terminate()
    print(f"Search time: {search_time}")
    print(f"Search time per query: {search_time/(numqueries)}")

    # all_scores = [scores_array, scores_array, ...]

    # all_scores = list(split(all_scores, params.num_threads))
    # all_indices = list(split(all_indices, params.num_threads))

    print("Filtering in Rust...")

    total_filtration_time = time.time()
    filtered_scores_list = my_rust_module.filter_scores(
        all_scores, all_indices, unrolled_names
    )
    # pool.map(filter_only, arg_list)
    # pool.terminate()

    total_filtration_time = time.time() - total_filtration_time
    elapsed_time = time.time() - start

    print(f"Filtration time: {total_filtration_time}.")
    print(f"Filtation time per query: {total_filtration_time/(numqueries)}")

    print(f"Elapsed time: {elapsed_time}.")

    print(f"Elapsed time per query: {elapsed_time/numqueries}.")


def evaluate(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")
    model_class = load_model_class(params.model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    target_embeddings, target_names, target_lengths = load_targets(
        params.target_embeddings,
        params.target_names,
        params.target_lengths,
        params.target_file,
        params.num_threads,
        model,
        params.max_seq_length,
        params.device,
    )

    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
        index_path=params.index_path,
    )

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data
    # query_sequences = {
    #    k: v
    #    for k, v in zip(
    #        list(query_sequences.keys())[:100], list(query_sequences.values())[:100]
    #    )
    # }

    duration, total_search_time, total_filtration_time = filter(
        [
            query_sequences,
            model,
            params.save_dir,
            index,
            unrolled_names,
            params.max_seq_length,
            params.write_results,
        ]
    )
    print(f"Duration: {duration}.")
    print(f"Search time: {total_search_time}.")
    print(f"Filtration time: {total_filtration_time}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()
    configfile = args.config.strip(".yaml")

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)
    if _config["num_threads"] > 1:
        evaluate_for_times_mp2(_config)
    else:
        evaluate(_config)
