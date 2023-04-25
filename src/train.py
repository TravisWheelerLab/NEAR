
import torch 
from sacred import Experiment 
import os
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from types import SimpleNamespace
from pytorch_lightning.loggers import TensorBoardLogger
from src.callbacks import CallbackSet
import time 
from src.train_config import *
import yaml 
from src.utils.util import (
    load_dataset_class,
    load_evaluator_class,
    load_model_class,
)

HOME = os.environ["HOME"]
train_ex = Experiment('train')


def train(_config):
    seed_everything(_config["seed"])
    params = SimpleNamespace(**_config)
    model_class = load_model_class(params.model_name)
    dataset_class = load_dataset_class(params.dataset_name)

    model = model_class(**params.model_args)
    train_dataset = dataset_class(**params.train_dataset_args)

    if hasattr(params, "val_dataset_args"):
        val_dataset = dataset_class(**params.val_dataset_args)
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
        save_dir=params.log_dir,
        name=params.model_name,
    )

    logger.experiment.add_text(
        tag="description", text_string=params.description, walltime=time.time()
    )

    trainer = Trainer(
        **params.trainer_args,
        callbacks=CallbackSet.callbacks(),
        logger=logger,
        val_check_interval=0.2,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':

    with open("src/configs/train.yaml", "r") as stream:
        _config = yaml.safe_load(stream)
    train(_config)