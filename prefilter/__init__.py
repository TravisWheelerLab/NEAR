"""
Prefilter passes good candidates to hmmer.
"""
import os
from argparse import ArgumentParser

__version__ = "0.0.1"

id_to_class_code = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "resources/accession_id_to_class_code.json",
)

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
    train_parser.add_argument("--gpus", type=int, required=True)
    train_parser.add_argument("--num_nodes", type=int, required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--normalize_output_embedding", action="store_true")
    train_parser.add_argument("--learning_rate", type=float, required=True)
    train_parser.add_argument("--batch_size", type=int, required=True)
    train_parser.add_argument("--num_workers", type=int, required=True)
    train_parser.add_argument("--evaluating", action="store_true")
    train_parser.add_argument("--check_val_every_n_epoch", type=int, required=True)
    train_parser.add_argument("--model_name", type=str, required=True)
    train_parser.add_argument("--data_path", type=str, required=True)
    train_parser.add_argument("--decoy_path", type=str, required=True)
    train_parser.add_argument("--tune_initial_lr", action="store_true")
    train_parser.add_argument("--schedule_lr", action="store_true")
    train_parser.add_argument("--step_lr_step_size", type=int, default=None)
    train_parser.add_argument("--step_lr_decay_factor", type=float, default=None)
    train_parser.add_argument("--min_unit", type=int, default=1)
    train_parser.add_argument("--res_block_n_filters", type=int, default=None)
    train_parser.add_argument("--vocab_size", type=int, default=None)
    train_parser.add_argument("--res_block_kernel_size", type=int, default=None)
    train_parser.add_argument("--n_res_blocks", type=int, default=None)
    train_parser.add_argument("--res_bottleneck_factor", type=float, default=None)
    train_parser.add_argument("--dilation_rate", type=float, default=None)
    train_parser.add_argument("--project_name", type=str, default="prefilter")
    train_parser.add_argument("--shoptimize", action="store_true")
    train_parser.add_argument("--log_confusion_matrix", action="store_true")
    train_parser.add_argument("--n_seq_per_fam", default=None)
    train_parser.add_argument("--emission_sequence_path", default=None)

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
