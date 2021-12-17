import tempfile
import pdb
import prefilter.utils as utils
import subprocess
import pandas as pd
import logging
import time
import os
from argparse import ArgumentParser
from label_fasta import parse_tblout, labels_from_file

log = logging.getLogger(__name__)


def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--train_fasta_file",
        type=str,
        help="fasta file containing the clustered, labeled train sequences",
    )
    parser.add_argument(
        "-a",
        "--alignment",
        type=str,
        help="(full) alignment containing sequences in train_fasta_file",
    )
    parser.add_argument(
        "-o",
        "--train_alignment_out_directory",
        type=str,
        help="where to save the training alignment",
    )
    parser.add_argument(
        "-r",
        "--relabeled_directory",
        type=str,
        help="directory containing labeled fasta files",
    )
    parser.add_argument("--evalue_threshold", type=float, default=1e-5)
    return parser.parse_args()


def reclassify(fasta_file, hmm_file, relabeled_directory):

    if not os.path.isfile(fasta_file):
        log.info(f"{fasta_file} does not exist")
        return

    if not os.path.isfile(hmm_file):
        log.info(f"{hmm_file} does not exist")
        return

    # reclassify the fasta file with the hmm:
    # wait. Do I want to reclassify the fasta file with all of the hmms from the training set?
    # probably... for now since we're doing single label classification this will do.

    tblout_path = os.path.splitext(fasta_file)[0] + ".tblout"

    relabeled_path = os.path.join(relabeled_directory, os.path.basename(fasta_file))
    subprocess.call(
        f"hmmsearch -o /dev/null --tblout {tblout_path} {hmm_file} {fasta_file}".split()
    )
    if os.path.isfile(tblout_path):
        tblout_df = parse_tblout(tblout_path)
        labels_from_file(fasta_file, relabeled_path, tblout_df, relabel=True)
    else:
        log.info(f"couldn't find tblout {tblout_path}; did hmmsearch work correctly?")


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
                raise ValueError(
                    f"has {args.train_fasta_file} been relabeled with "
                    f"label_fasta.py?"
                )

    ali_out_path = (
        os.path.join(
            args.train_alignment_out_directory,
            os.path.splitext(os.path.basename(args.alignment))[0],
        )
        + "-train.sto"
    )

    # extract the training alignment
    subprocess.call(
        f"esl-alimanip -o {ali_out_path} --seq-k {random_f} {args.alignment}".split()
    )
    os.remove(random_f)
    # build a new hmm
    hmm_out_path = os.path.splitext(ali_out_path)[0] + ".hmm"
    subprocess.call(f"hmmbuild -o /dev/null {hmm_out_path} {ali_out_path}".split())
    # then reclassify the training, test, and validation file with it (if they exist).
    test_file = os.path.join(
        os.path.dirname(args.train_fasta_file),
        os.path.basename(args.train_fasta_file).replace("train", "test"),
    )

    valid_file = os.path.join(
        os.path.dirname(args.train_fasta_file),
        os.path.basename(args.train_fasta_file).replace("train", "valid"),
    )

    reclassify(args.train_fasta_file, hmm_out_path, args.relabeled_directory)
    reclassify(test_file, hmm_out_path, args.relabeled_directory)
    reclassify(valid_file, hmm_out_path, args.relabeled_directory)
