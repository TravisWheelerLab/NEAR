"""
Prefilter passes good candidates to hmmer.
"""

__version__ = "0.0.1"

import os
import pdb
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning
import torch
from pytorch_lightning import seed_everything
from sacred.observers import FileStorageObserver

from src.callbacks import CallbackSet
from src.config import ex
from src.utils.util import load_dataset_class, load_model_class

ex.add_config({"testing": "am i working?"})


@ex.config
def _observer(log_dir, model_name):
    ex.observers.append(FileStorageObserver(f"{log_dir}/{model_name}/"))


@ex.config
def _cls_loader(model_name, dataset_name):
    model_class = load_model_class(model_name)
    dataset_class = load_dataset_class(dataset_name)


@ex.config
def _trainer_args(trainer_args):
    # set fairly permanent trainer args here.

    if trainer_args["gpus"] > 0:
        trainer_args["precision"] = 16

    trainer_args["strategy"] = "ddp"


@ex.main
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
        train_dataset, collate_fn=train_dataset.collate_fn(), **params.dataloader_args
    )

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, collate_fn=val_dataset.collate_fn(), **params.dataloader_args
        )
    else:
        val_dataloader = None

    logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=os.path.split(ex.observers[0].dir)[0],
        version=Path(ex.observers[0].dir).name,
        name="",
    )

    trainer = pytorch_lightning.Trainer(
        **params.trainer_args, callbacks=CallbackSet.callbacks(), logger=logger
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


ex.run()