import os
from pytorch_lightning import seed_everything
from time import time

seed_everything(int(time()) // 1000)
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from shopty import ShoptyConfig

from glob import glob
from argparse import ArgumentParser

from prefilter.models import Prot2Vec
from prefilter.utils import PROT_ALPHABET


def main(args):
    if args.schedule_lr and (
        args.step_lr_step_size is None or args.step_lr_decay_factor is None
    ):
        raise ValueError(
            "--schedule_lr requires --step_lr_step_size and --step_lr_decay_factor"
        )
    if args.train_from_scratch:
        # TODO: add error checking to make sure at least one of the
        # arguments is filled out for prot2vec
        pass

    if args.shoptimize:
        shopty_config = ShoptyConfig()
        result_file = shopty_config.results_path
        experiment_dir = shopty_config.experiment_directory
        checkpoint_dir = shopty_config.checkpoint_directory
        checkpoint_file = shopty_config.checkpoint_file
        max_iter = shopty_config.max_iter
        min_unit = 50
        print(
            "frog",
            result_file,
            experiment_dir,
            checkpoint_dir,
            checkpoint_file,
            max_iter,
        )
    else:
        max_iter = args.epochs

    data_path = args.data_path

    if "$HOME" in data_path:
        data_path = data_path.replace("$HOME", os.environ["HOME"])

    train_files = glob(os.path.join(data_path, "*train*"))
    val_files = glob(os.path.join(data_path, "*valid*"))
    decoy_files = glob(os.path.join(args.decoy_path, "*.fa"))

    data_and_optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "train_files": train_files,
        "val_files": val_files,
        "decoy_files": decoy_files,
        "schedule_lr": args.schedule_lr,
        "step_lr_step_size": args.step_lr_step_size,
        "step_lr_decay_factor": args.step_lr_decay_factor,
        "resample_families": args.resample_families,
        "resample_based_on_num_labels": args.resample_based_on_num_labels,
        "resample_based_on_uniform_dist": args.resample_based_on_uniform_dist,
        "train_from_scratch": args.train_from_scratch,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "single_label": args.single_label,
    }

    model = Prot2Vec(
        res_block_n_filters=args.res_block_n_filters,
        vocab_size=len(PROT_ALPHABET),
        res_block_kernel_size=args.res_block_kernel_size,
        n_res_blocks=args.n_res_blocks,
        res_bottleneck_factor=args.res_bottleneck_factor,
        dilation_rate=args.dilation_rate,
        **data_and_optimizer_kwargs,
    )

    if args.shoptimize:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=checkpoint_dir, save_last=True, save_top_k=0, verbose=True
        )
    else:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val/loss",
            filename="{epoch}-{val/loss:.5f}-{val/acc:.5f}",
            save_top_k=1,
        )

    last_epoch = 0

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

    log_lr = pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="step")

    trainer_kwargs = {
        "gpus": args.gpus,
        "num_nodes": args.num_nodes,
        "max_epochs": last_epoch + (max_iter * min_unit)
        if args.shoptimize
        else args.epochs,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "callbacks": [checkpoint_callback, log_lr],
        "accelerator": "ddp" if args.gpus else None,
        "plugins": DDPPlugin(find_unused_parameters=False),
        "precision": 16 if args.gpus else 32,
        "terminate_on_nan": True,
        "logger": pl.loggers.TensorBoardLogger(experiment_dir, name="", version="")
        if args.shoptimize
        else WandbLogger(
            save_dir=args.log_dir, log_model="all", project=args.project_name
        ),
    }

    if args.tune_initial_lr:
        trainer_kwargs["auto_lr_find"] = True
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.tune(model)
    else:
        trainer = pl.Trainer(**trainer_kwargs)

    ckpt_path = None
    if args.shoptimize:
        checkpoint_file = checkpoint_file if os.path.isfile(checkpoint_file) else None

    trainer.fit(model, ckpt_path=ckpt_path)
    # I use test as a handy override for doing things with the best model after training.
    if args.shoptimize:
        results = trainer.validate(model)[0]
        with open(result_file, "w") as dst:
            dst.write(f"test_acc:{results['val/loss']}")
    else:
        # this requires a wandb logger.
        results = trainer.validate(model)[0]

    if not args.shoptimize:
        torch.save(
            model.state_dict(),
            os.path.join(trainer.logger.experiment.dir, args.model_name),
        )
