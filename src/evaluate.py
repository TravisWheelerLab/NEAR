import torch
import os
import tqdm
from types import SimpleNamespace
import time
import yaml
import argparse
import itertools
from src.evaluators.contrastive_functional import (
    filter,
    _setup_targets_for_search,
    save_target_embeddings,
    _calc_embeddings,
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
import concurrent.futures

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


def load_targets(params):
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

    return target_embeddings, target_names, target_lengths


def evaluate_multiprocessing(_config):
    params = SimpleNamespace(**_config)

    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    model = load_model(params.checkpoint_path.params.model_name, params.device)

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")

    target_embeddings, target_names, target_lengths = load_targets(params)
    assert len(target_lengths) == len(target_names) == len(target_embeddings)

    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
    )

    query_names = list(query_sequences.keys())[:1000]

    sequences = list(query_sequences.values())[:1000]
    numqueries = len(query_names)
    print(f"Number of queries: {numqueries}")

    query_names = list(split(query_names, params.num_threads))
    assert len(query_names) == params.num_threads
    sequences = list(split(sequences, params.num_threads))
    arg_list = [
        (
            query_names[i],
            sequences[i],
            model,
            params.save_dir,
            index,
            unrolled_names,
            params.max_seq_length,
            params.write_results,
        )
        for i in range(params.num_threads)
    ]
    del query_sequences

    pool = Pool(params.num_threads)

    total_duration = 0
    total_search_time = 0
    total_filtration_time = 0

    start = time.time()

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #    futures = {executor.submit(filter, arg_list[i]) for i in range(len(arg_list))}

    #    for future in concurrent.futures.as_completed(futures):
    #        duration, search_time, filtration_time = future.result()
    #        total_duration += duration
    #        total_search_time += search_time
    #        total_filtration_time += filtration_time

    # total_duration = 0
    # total_search_time = 0
    # total_filtration_time = 0
    for result in pool.imap(filter, arg_list):
        duration, search_time, filtration_time = result
        total_duration += duration
        total_search_time += search_time
        total_filtration_time += filtration_time

    print(f"Duration per query: {duration / numqueries}.")
    print(
        f"CPU search time per query: {total_search_time / (params.num_threads * numqueries)}."
    )
    print(
        f"CPU filtration time per query: {total_filtration_time  / (params.num_threads * numqueries)}."
    )

    print(f"Elapsed time per query: {(time.time() - start) / numqueries}.")

    pool.terminate()


def search_only(query_data):
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


def evaluate_for_times_mp2(_config):
    params = SimpleNamespace(**_config)
    print(f"Index path: {params.index_path}")

    model_input = load_model(params.checkpoint_path, params.model_name, params.device)

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")

    target_embeddings, target_names, target_lengths = load_targets(params)
    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data

    query_data = list(query_sequences.values())[:5000]

    query_names_list = list(query_sequences.keys())[:5000]

    numqueries = len(query_data)
    print(f"Number of queries: {numqueries}")

    print(f"Nprobe: {params.nprobe}")
    print(f"num threads: {params.num_threads}")
    print(f"omp_num_threads: {params.omp_num_threads}")

    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, faiss_index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
        index_path=params.index_path,
    )

    global model, output_path, index

    model = model_input
    output_path = params.save_dir
    index = faiss_index

    pool = Pool(params.num_threads)

    print("Searching with FAISS...")
    start = time.time()

    all_scores = []
    all_indices = []

    query_data = list(split(query_data, params.num_threads))

    for result in pool.imap(search_only, query_data):
        scores, indices = result
        all_scores += scores
        all_indices += indices
    search_time = time.time() - start
    print(f"Search time per query: {search_time/(numqueries)}")
    pool.terminate()

    filter_threads = 4

    all_scores = list(split(all_scores, filter_threads))
    all_indices = list(split(all_indices, filter_threads))
    query_names = list(split(query_names_list, filter_threads))

    print("Saving to file")

    with open("/xdisk/twheeler/daphnedemekas/all_scores.txt", "w") as f:
        for score in all_scores:
            # score is an array of (seq_len, 1000)
            row = ""
            column = ""
            for value in score:
                row += f"{value[0]}, "
                column += f"{value[1]}, "
            f.write(f"{row} : {column}" + "\n")

    with open("/xdisk/twheeler/daphnedemekas/all_indices.txt", "w") as f:
        for ind in all_indices:
            # score is an array of (seq_len, 1000)
            row = ""
            column = ""
            for value in ind:
                row += f"{value[0]}, "
                column += f"{value[1]}, "
            f.write(f"{row} : {column}" + "\n")

    with open("/xdisk/twheeler/daphnedemekas/unrolled_names.txt", "w") as f:
        for name in unrolled_names:
            f.write(name + "\n")

    print("Saved")

    arg_list = [
        (
            all_scores[i],
            all_indices[i],
            unrolled_names,
            query_names[i],
            params.write_results,
            params.save_dir,
        )
        for i in range(filter_threads)
    ]

    pool = MPool(filter_threads)
    print(f"Filtering with {filter_threads} threads")
    filtration_time = time.time()

    pool.map(filter_only, arg_list)
    print(f"Filtration time per query: {(time.time() - filtration_time) / numqueries}")
    pool.terminate()
    #      query_names, scores, indices = result
    #       #    query_names_list += query_names
    #       all_scores += scores
    #       all_indices += indices
    #       query_names_list += query_names
    # search_time = time.time() - start
    print(f"Search time: {search_time}")
    print(f"Search time per query: {search_time/(numqueries)}")

    # all_scores = list(split(all_scores, params.num_threads))
    # all_indices = list(split(all_indices, params.num_threads))

    # print("Filtering in Rust...")

    # total_filtration_time = time.time()
    # filtered_scores_list = my_rust_module.filter_scores(
    #    all_scores, all_indices, unrolled_names
    # )
    # pool.map(filter_only, arg_list)
    # pool.terminate()

    # total_filtration_time = time.time() - total_filtration_time
    elapsed_time = time.time() - start

    # print(f"Filtration time: {total_filtration_time}.")
    # print(f"Filtation time per query: {total_filtration_time/(numqueries)}")

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
        # evaluate_multiprocessing(_config)
        evaluate_for_times_mp2(_config)
    else:
        evaluate(_config)
