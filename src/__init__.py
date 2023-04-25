#!/home/u4/colligan/venvs/prefilter/bin/python3
"""
Prefilter passes good candidates to hmmer.
"""

__version__ = "0.0.1"

import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import pdb

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred.observers import FileStorageObserver
from src.callbacks import CallbackSet
from src.eval_config import evaluation_ex
from src.train_config import train_ex
from src.utils.util import (
    load_dataset_class,
    load_evaluator_class,
    load_model_class,
)
from multiprocessing.pool import ThreadPool as Pool
torch.multiprocessing.set_sharing_strategy('file_system')

@train_ex.config
def _observer(log_dir, model_name):
    train_ex.observers.append(FileStorageObserver(f"{log_dir}/{model_name}/"))


@train_ex.config
def _cls_loader(model_name, dataset_name):
    model_class = load_model_class(model_name)
    dataset_class = load_dataset_class(dataset_name)


@train_ex.config
def _log_verbosity(log_verbosity):
    logger = logging.getLogger("train")
    logger.setLevel(log_verbosity)


@train_ex.config
def _trainer_args(trainer_args):
    # set fairly permanent trainer args here.
    if trainer_args["accelerator"] == "gpu":
        trainer_args["precision"] = 16
    # trainer_args["detect_anomaly"] = True


@train_ex.config
def _ensure_description(description):
    if description == "":
        if sys.stdout.isatty():
            description = input("Describe your experiment.")
        else:
            raise ValueError("Describe your experiment by editing train_config.py.")


@train_ex.main
def train(_config):
    seed_everything(_config["seed"])
    params = SimpleNamespace(**_config)
    model = params.model_class(**params.model_args)
    train_dataset = params.dataset_class(**params.train_dataset_args)

    if hasattr(params, "val_dataset_args"):
        val_dataset = params.dataset_class(**params.val_dataset_args)
    else:
        val_dataset = None

    print(f"Training model {params.model_name} with dataset {params.dataset_name}.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn(), **params.dataloader_args,
    )

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, collate_fn=val_dataset.collate_fn(), **params.dataloader_args,
        )
    else:
        val_dataloader = None

    logger = TensorBoardLogger(
        save_dir=os.path.split(train_ex.observers[0].dir)[0],
        version=Path(train_ex.observers[0].dir).name,
        name="",
    )

    logger.experiment.add_text(
        tag="description", text_string=params.description, walltime=time.time()
    )

    trainer = Trainer(
        **params.trainer_args,
        callbacks=CallbackSet.callbacks(),
        logger=logger,
        val_check_interval=0.2,
        resume_from_checkpoint = 'ResNet1dMultiPos/epoch_0_4.36850.cpkt'
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )


# test whether or not I'm interactive
@evaluation_ex.config
def _cls_loader(model_name, evaluator_name):
    model_class = load_model_class(model_name)
    evaluator_class = load_evaluator_class(evaluator_name)


@evaluation_ex.config
def _thread_exporter(num_threads):
    os.environ["NUM_THREADS"] = str(num_threads)


@evaluation_ex.config
def _log_verbosity(log_verbosity):
    logger = logging.getLogger("evaluate")
    logger.setLevel(log_verbosity)


def evaluate_multiprocessing(_config):
    import itertools
    from src.evaluators.uniref_evaluator import _calc_embeddings
    params = SimpleNamespace(**_config)

    params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")

    model = params.model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path, map_location=torch.device(params.device),
    ).to(params.device)


    evaluator = params.evaluator_class(**params.evaluator_args)
    evaluator.model_class = model

    query_sequences = evaluator.query_seqs
    params.logger.info("Splitting data into chunks")

    q_chunk_size = len(query_sequences) // params.num_threads

    target_embeddings = torch.load('target_embeddings.pt')
    print(len(target_embeddings))
    with open('target_names.txt','r') as f:
        target_names = f.readlines()
        target_names = [t.strip("\n") for t in target_names]
    with open('target_lengths.txt','r') as f:
        target_lengths = f.readlines()
        lengths = [int(t.strip("\n")) for t in target_lengths]
    
    assert len(lengths) == len(target_names) == len(target_embeddings)


    query_sequence_chunks = [(dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)), model) for i in range(0,len(query_sequences), q_chunk_size)]
    del query_sequences 

    pool = Pool(params.num_threads)

    query_names = []
    query_embeddings = []

    params.logger.info("Embedding queries...")

    start_time = time.time()

    for result in pool.imap(_calc_embeddings, query_sequence_chunks):
        n, e, _ = result
        query_names.extend(n)
        query_embeddings.extend(e)

    # assert len(query_names) == len(query_embeddings)

    loop_time = time.time() - start_time
    params.logger.info(f"Query embedding took: {loop_time}.")
    pool.terminate()
    evaluator.evaluate_multiprocessing(query_names, query_embeddings, target_names, target_embeddings, lengths)

    #result = evaluator.evaluate_multiprocessing(query_names, query_embeddings, target_names, target_embeddings, lengths)

def evaluate_multiprocessing2(_config):
    import itertools
    from src.evaluators.contrastive_functional import filter, _setup_targets_for_search
    params = SimpleNamespace(**_config)

    params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")

    model = params.model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path, map_location=torch.device(params.device),
    ).to(params.device)

    query_sequences = params.evaluator_args["query_seqs"]

    q_chunk_size = len(query_sequences) // params.num_threads

    target_embeddings = torch.load('target_embeddings.pt')
    with open('target_names.txt','r') as f:
        target_names = f.readlines()
        target_names = [t.strip("\n") for t in target_names]
    with open('target_lengths.txt','r') as f:
        target_lengths = f.readlines()
        lengths = [int(t.strip("\n")) for t in target_lengths]
    
    assert len(lengths) == len(target_names) == len(target_embeddings)
    unrolled_names, index = _setup_targets_for_search(target_embeddings, target_names, target_lengths, params.index_string, params.nprobe)

    arg_list = [(dict(itertools.islice(query_sequences.items(), i, i + q_chunk_size)), model, params.save_dir, index, unrolled_names) for i in range(0,len(query_sequences), q_chunk_size)]
    del query_sequences 

    #pool = Pool(params.num_threads)
    pool = Pool(params.num_threads)

    params.logger.info("Beginning search...")

    start_time = time.time()

    for result in pool.imap(filter, arg_list):
        print("Got result")

    loop_time = time.time() - start_time
    params.logger.info(f"Entire search took: {loop_time}.")
    pool.terminate()

@evaluation_ex.main
def evaluate(_config):

    params = SimpleNamespace(**_config)

    if "test" in params.save_dir:
        evaluate_multiprocessing2(_config)
    elif params.device == 'cpu' and params.num_threads > 1:
        evaluate_multiprocessing(_config)
    else:

        params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")

        model = params.model_class.load_from_checkpoint(
            checkpoint_path=params.checkpoint_path, map_location=torch.device(params.device),
        ).to(params.device)
        print("here")

        evaluator = params.evaluator_class(**params.evaluator_args)

        evaluator.evaluate(model_class=model)

def train_main():
    train_ex.run_commandline()


def evaluate_main():
    evaluation_ex.run_commandline()
