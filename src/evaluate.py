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
from src.evaluators.contrastive_functional import (
    _setup_targets_for_search,
    save_target_embeddings,
    _calc_embeddings,
    # search,
    search_and_filter,
)
from multiprocessing.pool import ThreadPool as Pool
import concurrent.futures

# from multiprocessing import Pool
from src.utils.util import (
    load_model_class,
)
import pickle
import my_rust_module
import numpy as np
from src.data.hmmerhits import FastaFile

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


def load_model(checkpoint_path, model_name, device="cpu"):
    print(f"Loading from checkpoint in {checkpoint_path}")

    model_class = load_model_class(model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device(device),
    ).to(device)

    return model


def load_index(params):
    if not os.path.exists(params.target_embeddings):
        print("No saved target embeddings. Calculating them now.")
        targetfasta = FastaFile(params.target_file)
        target_sequences = targetfasta.data

        target_names, target_lengths, target_embeddings = save_off_targets(
            target_sequences,
            params.num_threads,
            model,
            params.max_seq_length,
            params.device,
            params.target_embeddings,
        )
    else:
        target_embeddings = torch.load(params.target_embeddings)

        if params.target_names.endswith(".pickle"):
            with open(params.target_names, "rb") as file_handle:
                target_names = pickle.load(file_handle)

            with open(params.target_lengths, "rb") as file_handle:
                target_lengths = pickle.load(file_handle)

        elif params.target_names.endswith(".txt"):
            with open(params.target_names, "r") as f:
                target_names = f.readlines()
                target_names = [t.strip("\n") for t in target_names]
            with open(params.target_lengths, "r") as f:
                target_lengths = f.readlines()
                target_lengths = [int(t.strip("\n")) for t in target_lengths]

        else:
            raise Exception("Saved target data format not understood")

    assert (
        len(target_lengths) == len(target_names) == len(target_embeddings)
    ), "Target lengths, names and embeddings are not all the same length"

    index = _setup_targets_for_search(
        target_embeddings,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
        index_path=params.index_path,
    )

    return index


def save_FAISS_results(
    query_names,
    all_scores,
    all_indices,
    scores_path="/xdisk/twheeler/daphnedemekas/all_scores-reversed.h5",
    indices_path="/xdisk/twheeler/daphnedemekas/all_indices-reversed.h5",
    query_names_path="/xdisk/twheeler/daphnedemekas/query_names-reversed.txt",
):
    print("Saving FAISS results")

    with h5py.File(scores_path, "w") as hf:
        for i, arr in enumerate(all_scores):
            hf.create_dataset(f"array_{i}", data=arr)

    with h5py.File(indices_path, "w") as hf:
        for i, arr in enumerate(all_indices):
            hf.create_dataset(f"array_{i}", data=arr)

    with open(query_names_path, "w") as f:
        for name in query_names:
            f.write(name + "\n")
    print("Saved")


def search(args):
    (query_data, model, output_path, index_path, nprobe, max_seq_length) = args
    # index = faiss.read_index(index_path)
    index = index_path
    index.nprobe = nprobe
    query_names, queries, _ = _calc_embeddings(query_data, model, max_seq_length)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    all_scores = []
    all_indices = []

    print("Searching...")

    for i in range(len(queries)):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        all_scores.append(scores.to("cpu").numpy())

        #   all_indices.append(reduce_indices(indices.to("cpu").numpy(), index_mapping))
        all_indices.append(indices.to("cpu").numpy())
    index.close()
    return query_names, all_scores, all_indices


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
    q_chunk_size = len(query_sequences) // params.num_threads

    index = load_index(params)

    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            # params.index_path,
            index,
            params.nprobe,
            params.max_seq_length,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences
    print(f"Length of arg list: {len(arg_list)}")
    pool = Pool(params.num_threads)

    print("Beginning search...")
    start = time.time()
    # results = pool.map(search, arg_list)
    query_names_list = []
    all_scores_list = []
    all_indices_list = []
    for result in pool.imap(search, arg_list):
        query_names, all_scores, all_indices = result
        query_names_list += query_names
        all_scores_list += all_scores
        all_indices_list += all_indices

    print(f"Search time: {time.time() - start}.")

    pool.terminate()
    if params.write_results:
        save_FAISS_results(
            query_names_list,
            all_scores_list,
            all_indices_list,
            params.scores_path,
            params.indices_path,
            params.query_names_path,
        )


def evaluate_multiprocessing2(_config):
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
    q_chunk_size = len(query_sequences) // params.num_threads

    index = load_index(params)

    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            # params.index_path,
            index,
            params.nprobe,
            params.max_seq_length,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences
    print(f"Length of arg list: {len(arg_list)}")

    print("Beginning search...")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=params.num_threads
    ) as executor:
        future_to_batch = {executor.submit(search, batch): batch for batch in arg_list}
    query_names_list = []
    all_scores_list = []
    all_indices_list = []
    # Collect results as they become available
    for future in concurrent.futures.as_completed(future_to_batch):
        batch = future_to_batch[future]
        query_names, all_scores, all_indices = future.result()
        query_names_list += query_names
        all_scores_list += all_scores
        all_indices_list += all_indices  # ... combine results ...

    print(f"Search time: {time.time() - start}.")


def get_index_mapping(target_lengths):
    index_mapping = {}

    target_idx = 0
    j = 0
    for length in target_lengths:
        for i in range(length):
            k = i + j
            index_mapping[k] = target_idx
        j += length
        target_idx += 1
    return index_mapping


def search_only_new(query_data):
    global model, output_path, index
    # print(query_data[0])
    max_seq_length = 512
    # query_data = {q[0] : q[1] for q in query_data}
    queries, _ = _calc_embeddings(query_data, model, max_seq_length)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    all_scores = []
    all_indices = []

    for i in range(len(queries)):
        scores, indices = index.search(queries[i].contiguous(), k=1000)
        all_scores.append(scores.to("cpu").numpy())
        all_indices.append(indices.to("cpu").numpy())

    return all_scores, all_indices


def evaluate_multiprocessing_python(_config):
    params = SimpleNamespace(**_config)
    print(f"Index path: {params.index_path}")

    model = load_model(params.checkpoint_path, params.model_name, params.device)

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")

    index, _ = load_index(params)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    q_chunk_size = len(query_sequences) // params.num_threads

    numqueries = len(query_sequences)
    print(f"Number of queries: {numqueries}")

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")

    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            index_mapping,
            params.save_dir,
            index,
            params.max_seq_length,
            params.write_results,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences

    pool = Pool(params.num_threads)

    print("Beginning search...")
    start = time.time()

    query_names_list = []
    all_scores_list = []
    all_indices_list = []
    for result in pool.imap(search_and_filter, arg_list):
        query_names, all_scores, all_indices = result
        query_names_list += query_names
        all_scores_list += all_scores
        all_indices_list += all_indices

    print(f"Elapsed time: {time.time() - start}.")

    pool.terminate()


def evaluate(_config):
    params = SimpleNamespace(**_config)
    print("Why am i here")
    print(params.num_threads)
    print(f"Loading from checkpoint in {params.checkpoint_path}")

    model_class = load_model_class(params.model_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    index = load_index(params)

    arg_list = [
        query_sequences,
        model,
        #        index_mapping,
        params.save_dir,
        index,
        params.max_seq_length,
    ]
    # del query_sequences
    print(f"Number of queries: {len(query_sequences)}")
    del query_sequences
    print("Beginning search...")
    start = time.time()

    query_names, all_scores, all_indices = search(arg_list)

    print(f"Search time: {time.time() - start}.")

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
