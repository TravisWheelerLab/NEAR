"""
Prefilter passes good candidates to hmmer.
"""
from argparse import ArgumentParser

__version__ = "0.0.1"


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
    train_parser.add_argument("--resample_families", action="store_true")
    train_parser.add_argument("--resample_based_on_num_labels", action="store_true")
    train_parser.add_argument("--resample_based_on_uniform_dist", action="store_true")
    train_parser.add_argument("--tune_initial_lr", action="store_true")
    train_parser.add_argument("--schedule_lr", action="store_true")
    train_parser.add_argument("--step_lr_step_size", type=int, default=None)
    train_parser.add_argument("--step_lr_decay_factor", type=float, default=None)
    train_parser.add_argument("--min_unit", type=int, default=1)
    train_parser.add_argument("--train_from_scratch", action="store_true")
    train_parser.add_argument("--res_block_n_filters", type=int, default=None)
    train_parser.add_argument("--single_label", action="store_true")
    train_parser.add_argument("--vocab_size", type=int, default=None)
    train_parser.add_argument("--res_block_kernel_size", type=int, default=None)
    train_parser.add_argument("--n_res_blocks", type=int, default=None)
    train_parser.add_argument("--res_bottleneck_factor", type=float, default=None)
    train_parser.add_argument("--dilation_rate", type=float, default=None)
    train_parser.add_argument("--project_name", type=str, default="prefilter")
    train_parser.add_argument("--shoptimize", action="store_true")

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

    data_parser = subparsers.add_parser("data", help="data runner")

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
