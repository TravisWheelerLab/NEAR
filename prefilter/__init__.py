"""
Prefilter passes good candidates to hmmer.
"""
import os
from argparse import ArgumentParser

__version__ = "0.0.1"

import yaml

name_to_accession_id = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "resources/name_to_pfam_accession_id.yaml",
)


class PfamNameToAccessionID:

    name_to_accession_id = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resources/name_to_pfam_accession_id.yaml",
    )
    init = False

    def __getitem__(self, item):
        if self.init:
            return self.mapping[item]
        else:
            with open(self.name_to_accession_id, "r") as src:
                self.mapping = yaml.safe_load(src)
            self.init = True
            return self.mapping[item]


class AccessionIDToPfamName:

    name_to_accession_id = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resources/name_to_pfam_accession_id.yaml",
    )
    init = False

    def __getitem__(self, item):
        if self.init:
            return self.mapping[item]
        else:
            with open(self.name_to_accession_id, "r") as src:
                self.mapping = yaml.safe_load(src)
            self.mapping = {v: k for k, v in self.mapping.items()}
            self.init = True
            return self.mapping[item]


MASK_FLAG = -1
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
    train_parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help="list of specific GPUs to use with --specify_gpus set",
    )
    train_parser.add_argument("--num_nodes", type=int, required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--learning_rate", type=float, required=True)
    train_parser.add_argument("--batch_size", type=int, required=True)
    train_parser.add_argument("--num_workers", type=int, required=True)
    train_parser.add_argument("--check_val_every_n_epoch", default=2, type=int)
    train_parser.add_argument("--data_path", type=str, required=True)
    train_parser.add_argument("--logo_path", type=str, required=True)
    train_parser.add_argument("--decoy_path", type=str, default=None)
    train_parser.add_argument("--debug", action="store_true")
    train_parser.add_argument("--emission_path", nargs="+", type=str, default=None)
    train_parser.add_argument("--specify_gpus", action="store_true")
    train_parser.add_argument("--all_vs_all_loss", action="store_true")
    train_parser.add_argument("--supcon", action="store_true")

    # evaluation parser .----------------------------------------------------
    eval_parser = subparsers.add_parser("eval", help="evaluate a model")
    eval_parser.add_argument("--save_prefix", required=True)
    eval_parser.add_argument("--logs_dir", required=True)
    eval_parser.add_argument("--model_path", default=None)
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument(
        "--decoy_path",
        type=str,
        default="/home/tc229954/data/prefilter/small-dataset/random_sequences/random_sequences.fa",
    )

    args = ap.parse_args()
    if args.subcmd == "train":
        from prefilter.train import main

        main(args)
    elif args.subcmd == "eval":
        from prefilter.evaluate import main

        main(args)
    else:
        ap.print_help()
        exit(1)
