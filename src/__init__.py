#!/home/u4/colligan/venvs/prefilter/bin/python3
"""
Prefilter passes good candidates to hmmer.
"""

__version__ = "0.0.1"

import logging
import os
import pdb
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from sacred.observers import FileStorageObserver

from src.callbacks import CallbackSet
from src.eval_config import evaluation_ex
from src.train_config import train_ex
from src.utils.util import (
    load_dataset_class,
    load_evaluator_class,
    load_model_class,
)


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
        train_dataset,
        collate_fn=train_dataset.collate_fn(),
        **params.dataloader_args,
    )

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=val_dataset.collate_fn(),
            **params.dataloader_args,
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
        **params.trainer_args, callbacks=CallbackSet.callbacks(), logger=logger
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
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


@evaluation_ex.main
def evaluate(_config):
    params = SimpleNamespace(**_config)

    params.logger.info(f"Loading from checkpoint in {params.checkpoint_path}")

    model = params.model_class.load_from_checkpoint(
        checkpoint_path=params.checkpoint_path,
        map_location=torch.device(params.device),
    ).to(params.device)

    evaluator = params.evaluator_class(**params.evaluator_args)

    result = evaluator.evaluate(model_class=model)
    # now, track output of this evaluation with DVC.


def train_main():
    train_ex.run_commandline()


def evaluate_main():
    evaluation_ex.run_commandline()
