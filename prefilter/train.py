# pylint: disable=no-member
import os
import pdb
import random

from pytorch_lightning import seed_everything
from time import time

# seed_everything(20943)

import torch
import pytorch_lightning as pl
from random import shuffle, seed
from shopty import ShoptyConfig

from glob import glob
from argparse import ArgumentParser

from prefilter.models import ResNet1d
import prefilter.utils as utils


def main(args):
    pdb.set_trace()

    if args.msa_transformer:
        afa_files = glob(
            "/home/tc229954/data/prefilter/pfam/seed/20piddata/train/*afa"
        )[:1000]
        train_dataset = utils.MSAGenerator(afa_files=afa_files)
        valid_dataset = utils.MSAGenerator(
            afa_files=afa_files,
        )
        collate_fn = utils.msa_transformer_collate(
            with_labelvectors=args.only_aligned_characters
        )
    else:
        train_dataset = utils.SwissProtGenerator(
            fa_file="/home/tc229954/data/prefilter/uniprot/uniprot_sprot.fasta",
            minlen=args.seq_len,
        )
        valid_dataset = utils.SwissProtGenerator(
            fa_file="/home/tc229954/data/prefilter/uniprot/uniprot_sprot.fasta",
            training=False,
            minlen=args.seq_len,
        )
        collate_fn = utils.pad_contrastive_batches

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = ResNet1d(
        learning_rate=args.learning_rate,
        embed_msas=args.msa_transformer,
        apply_attention=args.apply_attention,
    )

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="epoch_{epoch}_{val_loss:.6f}",
        auto_insert_metric_name=False,
        save_top_k=500,
    )

    trainer_kwargs = {
        "gpus": args.gpus,
        "num_nodes": args.num_nodes,
        "max_epochs": args.epochs,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "callbacks": [checkpoint_callback],
        "precision": 16 if args.gpus else 32,
        "logger": pl.loggers.TensorBoardLogger(args.log_dir),
        "strategy": "ddp",
    }

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
