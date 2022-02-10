# pylint: disable=no-member
import os
from pytorch_lightning import seed_everything
from time import time

seed = 1639426
seed_everything(seed)
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from random import shuffle
from shopty import ShoptyConfig

from glob import glob
from argparse import ArgumentParser

from prefilter.models import Prot2Vec
from prefilter.utils import PROT_ALPHABET, create_class_code_mapping


def main(args):
    if args.schedule_lr and (
        args.step_lr_step_size is None or args.step_lr_decay_factor is None
    ):
        raise ValueError(
            "--schedule_lr requires --step_lr_step_size and --step_lr_decay_factor"
        )

    # shopty-specific config
    if args.shoptimize:
        shopty_config = ShoptyConfig()
        result_file = shopty_config.results_path
        experiment_dir = shopty_config.experiment_directory
        checkpoint_dir = shopty_config.checkpoint_directory
        checkpoint_file = shopty_config.checkpoint_file
        max_iter = shopty_config.max_iter
        min_unit = 1
    else:
        max_iter = args.epochs

    data_path = args.data_path
    emission_sequence_path = args.emission_sequence_path

    if "$HOME" in data_path:
        data_path = data_path.replace("$HOME", os.environ["HOME"])

    # select a subset of files to train on
    train_files = glob(os.path.join(data_path, "*train.fa"))

    if not (len(train_files)):
        raise ValueError("no train files")

    shuffle(train_files)

    val_files = []
    # then get their corresponding validation files (we don't want to validate on a set of files that don't have a
    # relationship to the train files)
    for f in train_files:
        valid = f.replace("-train", "-valid")
        if os.path.isfile(valid):
            val_files.append(valid)

    if not (len(val_files)):
        raise ValueError("no valid files")

    # check if the user specified an emission sequence path, and grab the emission sequences generated from the same HMM
    # as our train sequences
    if emission_sequence_path is not None:
        if "$HOME" in emission_sequence_path:
            emission_sequence_path = emission_sequence_path.replace(
                "$HOME", os.environ["HOME"]
            )
        emission_files = glob(os.path.join(emission_sequence_path, "*fa"))
        if not len(emission_files):
            raise ValueError(f"no emission files found at {emission_sequence_path}")

    # create an overall class code mapping.
    # This is done on each training run. The alternative is keeping a shared mapping of name to class code but this can
    # waste a bunch of compute. For example, if you're training on a small subset of files you only want to classify the
    # labels that appear in the small subset.

    if emission_sequence_path is not None:
        name_to_class_code = create_class_code_mapping(
            train_files + val_files + emission_files
        )
    else:
        name_to_class_code = create_class_code_mapping(train_files + val_files)

    print("=========")
    print(len(name_to_class_code))
    print("=========")

    # TODO: remove decoy file stuff
    decoy_files = glob(os.path.join(args.decoy_path, "*.fa"))

    # these options are ingested into the class that all models should subclass. They control how the dataset
    # behaves and what variables to log.

    data_and_optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "train_files": train_files,
        "emission_files": emission_files
        if emission_sequence_path is not None
        else None,
        "val_files": val_files,
        "decoy_files": decoy_files,
        "schedule_lr": args.schedule_lr,
        "step_lr_step_size": args.step_lr_step_size,
        "step_lr_decay_factor": args.step_lr_decay_factor,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "log_confusion_matrix": args.log_confusion_matrix,
        "n_seq_per_fam": args.n_seq_per_fam,
        "name_to_class_code": name_to_class_code,
        "n_emission_sequences": args.n_emission_sequences,
    }
    # set up the model
    model = Prot2Vec(
        res_block_n_filters=args.res_block_n_filters,
        vocab_size=len(PROT_ALPHABET),
        res_block_kernel_size=args.res_block_kernel_size,
        n_res_blocks=args.n_res_blocks,
        res_bottleneck_factor=args.res_bottleneck_factor,
        dilation_rate=args.dilation_rate,
        fcnn=args.fcnn,
        pos_weight=args.pos_weight,
        **data_and_optimizer_kwargs,
    )

    # create the checkpoint callbacks (shopty requires the one name "checkpoint callback")
    early_stopping_callback = None
    if args.shoptimize:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=checkpoint_dir, save_last=True, save_top_k=0, verbose=True
        )
        best_loss_ckpt = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val/loss",
            filename="{epoch}-{val/loss:.5f}-{val/acc:.5f}",
            save_top_k=1,
        )
    else:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            filename="epoch_{epoch}-val_loss_{val/loss:.5f}_val_f1_{val/f1:.5f}",
            auto_insert_metric_name=False,
            save_top_k=50,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="val/f1",
            mode="max",
            min_delta=0.0024,
            patience=300,
        )

    last_epoch = 0
    # if we're optimizing, check for a checkpoint and load it if it exists.
    if args.shoptimize:
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            last_epoch = checkpoint["epoch"]
            model = model.load_from_checkpoint(
                checkpoint_file,
                map_location=torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu"),
            )

    # callback to monitor learning rate in tensorboard
    log_lr = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="step")
    gpus = args.gpus
    if args.specify_gpus:
        # consider the input as a LIST
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
        "max_epochs": last_epoch + (max_iter * min_unit)
        if args.shoptimize
        else args.epochs,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "callbacks": [checkpoint_callback, log_lr, best_loss_ckpt]
        if args.shoptimize
        else [checkpoint_callback, log_lr, early_stopping_callback],
        "accelerator": "ddp" if args.gpus else None,
        "plugins": DDPPlugin(find_unused_parameters=False),
        "precision": 16 if args.gpus else 32,
        "terminate_on_nan": True,
        "logger": pl.loggers.TensorBoardLogger(experiment_dir, name="", version="")
        if args.shoptimize
        else pl.loggers.TensorBoardLogger(args.log_dir),
    }
    if args.tune_initial_lr:
        trainer_kwargs["auto_lr_find"] = True
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.tune(model)
    else:
        trainer = pl.Trainer(**trainer_kwargs)

    ckpt_path = None
    if args.shoptimize:
        ckpt_path = checkpoint_file if os.path.isfile(checkpoint_file) else None

    # finally, train the model
    trainer.fit(model, ckpt_path=ckpt_path)

    if args.shoptimize:
        results = trainer.validate(model)[0]
        with open(result_file, "w") as dst:
            dst.write(f"test_acc:{results['val/loss']}")
