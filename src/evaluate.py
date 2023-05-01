import torch
import os
from types import SimpleNamespace
import time
import yaml
import argparse
import itertools
from src.evaluators.contrastive_functional import filter, _setup_targets_for_search
from src.data.hmmerhits import FastaFile
from src.data.eval_utils import get_evaluation_data
from multiprocessing.pool import ThreadPool as Pool

HOME = os.environ["HOME"]


def evaluate_multiprocessing(_config):

    params = SimpleNamespace(**_config)

    params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")

    model = params.model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    queryfasta = FastaFile(params.queryfile)
    query_sequences = queryfasta.data
    q_chunk_size = len(query_sequences) // params.num_threads

    target_embeddings = torch.load(params.target_embeddings)

    with open(params.target_names, "r") as f:
        target_names = f.readlines()
        target_names = [t.strip("\n") for t in target_names]
    with open(params.target_lengths, "r") as f:
        target_lengths = f.readlines()
        lengths = [int(t.strip("\n")) for t in target_lengths]

    assert len(lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(
        target_embeddings, target_names, target_lengths, params.index_string, params.nprobe
    )

    arg_list = [
        (
            dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)),
            model,
            params.save_dir,
            index,
            unrolled_names,
        )
        for i in range(0, len(query_sequences), q_chunk_size)
    ]
    del query_sequences

    pool = Pool(params.num_threads)

    params.logger.info("Beginning search...")

    start_time = time.time()

    for result in pool.imap(filter, arg_list):
        print("Got result")

    loop_time = time.time() - start_time
    params.logger.info(f"Entire search took: {loop_time}.")
    pool.terminate()


def evaluate(_config):

    params = SimpleNamespace(**_config)

    params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")

    model = params.model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)
    query_sequences, target_sequences, hmmer_hits_max = get_evaluation_data(
        params.query_file, params.evaluator_args["output_path"]
    )
    params.evaluator_args["query_seqyebces"] = query_sequences
    params.evaluator_args["target_sequences"] = target_sequences
    params.evaluator_args["hmmer_hits_max"] = hmmer_hits_max

    evaluator = params.evaluator_class(**params.evaluator_args)

    evaluator.evaluate(model_class=model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()
    configfile = args.config.strip(".yaml")

    with open(f"src/configs/{configfile}.yaml", "r") as stream:
        _config = yaml.safe_load(stream)

    if _config.num_threads > 1:
        evaluate_multiprocessing(_config)
    else:
        evaluate(_config)
