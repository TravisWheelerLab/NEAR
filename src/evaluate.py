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
from src.utils.util import (
    load_dataset_class,
    load_evaluator_class,
    load_model_class,
)
import pickle

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


def profile(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")
    model_class = load_model_class(params.model_name)
    evaluator_class = load_evaluator_class(params.evaluator_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)
    queryfasta = FastaFile(params.query_file)
    query_sequences = queryfasta.data
    if params.num_threads > 1:
        q_chunk_size = len(query_sequences) // params.num_threads
        names, sequences, lengths = embed_multithread(
            query_sequences, model, q_chunk_size
        )

    else:
        names, sequences, lengths = profile_embeddings(query_sequences, model, 512)


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

    q_chunk_size = len(query_sequences) // params.num_threads

    # get target embeddings
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

    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
    )

    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            index,
            unrolled_names,
            params.max_seq_length,
            params.write_results,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences

    pool = Pool(params.num_threads)

    print("Beginning search...")
    start = time.time()

    duration = 0
    for result in pool.imap(filter, arg_list):
        duration += result

    print(f"Total CPU time: {duration}.")
    print(f"Elapsed time: {time.time() - start}.")

    pool.terminate()


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

    # get target embeddings
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

    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.omp_num_threads,
    )
    print("Beginning search...")
    start = time.time()

    all_scores, all_indices, query_names, search_time = search_only(
        query_sequences, model, params.max_seq_length, index, params.output_path
    )

    arg_list = [
        (
            all_scores[i],
            all_indices[i],
            unrolled_names,
            params.write_results,
            params.output_path,
            query_names[i],
        )
        for i in range(len(all_scores))
    ]
    del query_sequences

    pool = Pool(params.num_threads)

    per_query_filtration_time = time.time()

    filtration_time = 0
    for result in pool.imap(filter_only, arg_list):
        filtration_time += result

    per_query_filtration_time = time.time() - per_query_filtration_time

    print(f"Elapsed time: {time.time() - start}.")
    print(f"FAISS parallelized Search time: {search_time}")
    print(f"Total filtration time: {filtration_time}.")
    print(f"Elapsed filtration time: {per_query_filtration_time}.")

    pool.terminate()


def evaluate(_config):
    params = SimpleNamespace(**_config)

    print(f"Loading from checkpoint in {params.checkpoint_path}")
    model_class = load_model_class(params.model_name)
    evaluator_class = load_evaluator_class(params.evaluator_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)
    query_sequences, target_sequences, hmmer_hits_max = get_evaluation_data(
        params.query_file, params.evaluator_args["output_path"]
    )
    params.evaluator_args["query_seqs"] = query_sequences
    params.evaluator_args["target_seqs"] = target_sequences
    params.evaluator_args["hmmer_hits_max"] = hmmer_hits_max

    evaluator = evaluator_class(**params.evaluator_args)

    evaluator.evaluate(model_class=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()
    configfile = args.config.strip(".yaml")

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)
    if _config["profiler"]:
        profile(_config)
    elif _config["num_threads"] > 1:
        evaluate_multiprocessing2(_config)
    else:
        evaluate(_config)
