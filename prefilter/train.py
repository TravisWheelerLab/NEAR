# pylint: disable=no-member
import os
from pytorch_lightning import seed_everything
from time import time

seed = 16394
seed_everything(seed)
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from random import shuffle
from shopty import ShoptyConfig

from glob import glob
from argparse import ArgumentParser

from prefilter.models import Prot2Vec, ResNet1d
from prefilter.utils import PROT_ALPHABET, create_class_code_mapping


def main(args):

    data_path = args.data_path

    if "$HOME" in data_path:
        data_path = data_path.replace("$HOME", os.environ["HOME"])

    train_files = glob(os.path.join(data_path, "*train.fa"))
    if args.debug:
        train_files = train_files[:3]

    if args.decoy_path is not None:
        decoy_files = glob(os.path.join(args.decoy_path, "*train.fa"))
        if not len(decoy_files):
            raise ValueError("no decoy files")

    if not (len(train_files)):
        raise ValueError("no train files")

    shuffle(train_files)

    # check if the user specified an emission sequence path, and grab the emission sequences generated from the same HMM
    # as our train sequences

    if args.emission_path is not None:
        emission_files = []
        for emission_sequence_path in args.emission_path:
            if "$HOME" in emission_sequence_path:
                emission_sequence_path = emission_sequence_path.replace(
                    "$HOME", os.environ["HOME"]
                )
            emission_files.extend(glob(os.path.join(emission_sequence_path, "*fa")))
            if not len(emission_files):
                raise ValueError(f"no emission files found at {emission_sequence_path}")
            if args.debug:
                emission_files = emission_files[:2]
                break

    val_files = []

    if args.emission_path is not None:
        name_to_class_code = create_class_code_mapping(
            train_files + val_files + emission_files
        )
    else:
        name_to_class_code = create_class_code_mapping(train_files + val_files)

    #
    model = ResNet1d(
        fasta_files=train_files,
        logo_path=args.logo_path,
        name_to_class_code=name_to_class_code,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        emission_files=emission_files if args.emission_path is not None else None,
        oversample_neighborhood_labels=False,
        num_workers=args.num_workers,
    )

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        filename="epoch_{epoch}_{train_loss}",
        auto_insert_metric_name=False,
        save_top_k=-1,
    )

    log_lr = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="step")
    gpus = args.gpus

    if args.specify_gpus:
        if not isinstance(gpus, list):
            gpus = [gpus]
    else:
        if len(gpus) != 1:
            raise ValueError(
                "Set --specify_gpus if you want to target training to a specific set of GPUs."
            )
        else:
            gpus = gpus[0]

    # create the arguments for the trainer
    trainer_kwargs = {
        "gpus": gpus,
        "num_nodes": args.num_nodes,
        "max_epochs": args.epochs,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "callbacks": [checkpoint_callback, log_lr],
        "precision": 16 if args.gpus else 32,
        "logger": pl.loggers.TensorBoardLogger(args.log_dir),
        "accelerator": "ddp",
        "plugins": DDPPlugin(find_unused_parameters=False),
    }

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model)
