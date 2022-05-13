"""
Prefilter passes good candidates to hmmer.
"""
import os
from argparse import ArgumentParser

__version__ = "0.0.1"

MASK_FLAG = -1
DROP_FLAG = -2
DECOY_FLAG = "DECOY"

array_job_template = """#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --output=array_out.out
#SBATCH --error=ERR
#SBATCH --nodes=1
#SBATCH --array=[1-ARRAY_JOBS]%200
DEPENDENCY
#SBATCH --cpus-per-task=1
#SBATCH --exclude=compute-1-11

f=$(sed -n "$SLURM_ARRAY_TASK_ID"p ARRAY_INPUT_FILE)
echo $f
RUN_CMD
"""
single_job_template = """#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --output=single_job_out.out
#SBATCH --nodes=1
DEPENDENCY
#SBATCH --cpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=compute-1-11

RUN_CMD
"""


def main():
    ap = ArgumentParser()
    ap.add_argument("--version", action="version", version="0.0.1")
    subparsers = ap.add_subparsers(title="subcommands", dest="subcmd")
    # train parser ---------------------------------------------------
    train_parser = subparsers.add_parser("train", help="train a model")
    train_parser.add_argument("--log_dir", required=True)
    train_parser.add_argument("--gpus", type=int)
    train_parser.add_argument("--num_nodes", type=int, required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--learning_rate", type=float, required=True)
    train_parser.add_argument("--batch_size", type=int, required=True)
    train_parser.add_argument("--num_workers", type=int, required=True)
    train_parser.add_argument("--afa_path", type=str, required=True)
    train_parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    train_parser.add_argument("--distill", action="store_true")
    train_parser.add_argument(
        "--use_embedding_layer_from_transformer", action="store_true"
    )
    train_parser.add_argument("--apply_mlp", action="store_true")
    train_parser.add_argument("--apply_substitutions", action="store_true")
    train_parser.add_argument("--embed_real_within_generated", action="store_true")

    args = ap.parse_args()
    if args.subcmd == "train":
        from prefilter.train import main

        main(args)
    else:
        ap.print_help()
        exit(1)
