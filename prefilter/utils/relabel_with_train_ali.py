import tempfile
import pdb
import prefilter.utils as utils
import subprocess
import pandas as pd
import logging
import time
import os
from argparse import ArgumentParser

def create_parser():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train_fasta_file", type=str,
                        help="fasta file containing the clustered, labeled train sequences")
    parser.add_argument("-a", "--alignment", type=str,
                        help="(full) alignment containing sequences in train_fasta_file")
    parser.add_argument("-o", "--train_alignment_out_directory", type=str,
                        help="where to save the training alignment")
    parser.add_argument("-r", "--relabeled_directory", type=str,
                        help="directory containing labeled fasta files")
    parser.add_argument("--evalue_threshold", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    # 1). Ingest a training fasta file
    # 2). Grab the sequences present in the train file from the .sto MSA.
    # 3). hmmbuild on the train file's sequences.
    # 4). Re-label the train, test, and validation fasta files with the new hmm.
    args = create_parser()
    if not os.path.isfile(args.train_fasta_file):
        # TODO: Refactor this to be an action in the argparser
        raise ValueError(f"{args.train_fasta_file} does not exist")

    if not os.path.isfile(args.alignment):
        # TODO: Refactor this to be an action in the argparser
        raise ValueError(f"{args.alignment} does not exist")

    os.makedirs(args.relabeled_directory, exist_ok=True)
    os.makedirs(args.train_alignment_out_directory, exist_ok=True)

    # grab the sequences in the seed alignment that are in the train set
    train_seq, _ = utils.fasta_from_file(args.train_fasta_file)
    random_f = f"/tmp/{str(time.time())}"
    with open(random_f, "w") as dst:
        for seq in train_seq:
            delim = seq.find(" |")
            if delim != -1:
                seq_name = seq[:delim].split()[0]
                dst.write(f"{seq_name}\n")
            else:
                raise ValueError(f"has {args.train_fasta_file} been relabeled with "
                                 f"label_fasta.py?")

    ali_out_path = os.path.join(args.train_alignment_out_directory,
                                os.path.splitext(os.path.basename(args.alignment))[0]) + "-train.sto"

    # extract the training alignment
    subprocess.call(f"esl-alimanip -o {ali_out_path} --seq-k {random_f} {args.alignment}".split())
    os.remove(random_f)
    # build a new hmm
    hmm_out_path = os.path.splitext(ali_out_path)[0] + ".hmm"
    subprocess.call(f"hmmbuild {hmm_out_path} {ali_out_path}".split())
    # then reclassify the training file with it


