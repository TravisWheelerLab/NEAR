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
)
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


def save_off_targets(target_sequences, num_threads, model, max_seq_length, model_device):

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

    torch.save(target_embeddings, "targets_2000.pt")
    with open("targetnames2000.pickle", "wb") as handle:
        pickle.dump(target_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("target_lengths_2000.pickle", "wb") as handle:
        pickle.dump(target_lengths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loop_time = time.time() - start_time
    print(f"Embedding took: {loop_time}.")


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

    if not os.path.exists(params.target_names):
        print("No saved target embeddings. Calculating them now.")
        targetfasta = FastaFile(params.target_file)
        target_sequences = targetfasta.data
        save_off_targets(
            target_sequences, params.num_threads, model, params.max_seq_length, params.device
        )

    target_embeddings = torch.load(params.target_embeddings)

    with open(params.target_names, "rb") as file_handle:
        target_names = pickle.load(file_handle)

    with open(params.target_lengths, "rb") as file_handle:
        target_lengths = pickle.load(file_handle)

    # with open(params.target_names, "r") as f:
    #     target_names = f.readlines()
    #     target_names = [t.strip("\n") for t in target_names]
    # with open(params.target_lengths, "r") as f:
    #     target_lengths = f.readlines()
    #     lengths = [int(t.strip("\n")) for t in target_lengths]

    assert len(target_lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings,
        target_names,
        target_lengths,
        params.index_string,
        params.nprobe,
        params.num_threads // 4,
    )

    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            index,
            unrolled_names,
            params.max_seq_length,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences

    pool = Pool(params.num_threads)

    print("Beginning search...")

    start_time = time.time()

    for result in pool.imap(filter, arg_list):
        print("Got result")

    loop_time = time.time() - start_time
    print(f"Entire search took: {loop_time}.")
    pool.terminate()


def evaluate(_config):

    params = SimpleNamespace(**_config)

    params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")
    model_class = load_model_class(params.model_name)
    evaluator_class = load_evaluator_class(params.evaluator_name)

    model = model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)
    query_sequences, target_sequences, hmmer_hits_max = get_evaluation_data(
        params.query_file, params.evaluator_args["output_path"]
    )
    params.evaluator_args["query_seqyebces"] = query_sequences
    params.evaluator_args["target_sequences"] = target_sequences
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

    if _config["num_threads"] > 1:
        evaluate_multiprocessing(_config)
    else:
        evaluate(_config)
