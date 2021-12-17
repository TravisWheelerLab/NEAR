#!/usr/bin/env python3
import pdb
import subprocess
import pandas as pd
import logging
import os

log = logging.getLogger(__name__)

from argparse import ArgumentParser
import prefilter.utils as utils

# TODO: replace hardcoded stuff with variables in __init__.py in prefilter

TBLOUT_COL_NAMES = [
    "target_name",
    "query_name",
    "accession_id",
    "e_value",
    "description",
]
TBLOUT_COLS = [0, 2, 3, 4, 18]


def parse_tblout(tbl):
    df = pd.read_csv(
        tbl,
        skiprows=3,
        header=None,
        delim_whitespace=True,
        usecols=TBLOUT_COLS,
        names=TBLOUT_COL_NAMES,
    )
    df["target_name"] = df["target_name"] + " " + df["description"]
    return df


def create_parser():
    parser = ArgumentParser()
    parser.add_argument("-a", "--aligned_fasta_file", type=str)
    parser.add_argument(
        "-p",
        "--pid",
        type=float,
        help="percent identity to split the aligned fasta file",
    )
    parser.add_argument(
        "-t",
        "--tblout",
        default=None,
        help="tblout containing the results of hmmsearch HMMDB aligned_fasta_file."
        " default: basename(aligned_fasta_file) + .tblout",
    )
    parser.add_argument(
        "-hdb",
        "--hmmdb",
        default=None,
        help="hmm database to search the aligned fasta file against (if the --tblout file doesn't exist",
    )
    parser.add_argument(
        "-c",
        "--carbs_output_directory",
        type=str,
        help="where to save the clustered .fa files.",
    )
    parser.add_argument(
        "-f",
        "--fasta_output_directory",
        type=str,
        help="where to save the labeled fasta files",
    )
    parser.add_argument("--evalue_threshold", type=float, default=1e-5)
    return parser.parse_args()


def labels_from_file(
    fasta_in, fasta_out, tblout_df, evalue_threshold=1e-5, relabel=False
):

    labels, sequences = utils.fasta_from_file(fasta_in)

    with open(fasta_out, "w") as dst:
        for label, sequence in zip(labels, sequences):
            if " |" in label:
                target_label = label[: label.find(" |")]
            else:
                target_label = label

            assigned_labels = tblout_df.loc[tblout_df["target_name"] == target_label]

            if len(assigned_labels) == 0:
                # why are some sequences not classified? They're in Pfam-seed,
                # which means they're manually curated to be part of a family.
                log.info(
                    f"sequence named {target_label} not found in {fasta_in} tblout."
                )
                continue
            # each sequence should have at least one label, but we
            # only need to grab one since one sequence can be associated with
            # multiple pfam accession IDs
            if relabel:
                # if we're relabelng, forget about adding a delimiter
                fasta_header = f">{label} "
            else:
                # otherwise, add it in.
                fasta_header = f">{label} | "

            labelset = []

            for seq_label, e_value in zip(
                assigned_labels["accession_id"], assigned_labels["e_value"]
            ):

                if float(e_value) <= evalue_threshold:
                    if relabel:
                        labelset.append("RL" + seq_label)
                    else:
                        labelset.append(seq_label)

            if len(labelset):
                fasta_header += " ".join(labelset) + "\n" + sequence + "\n"
                dst.write(fasta_header)


if __name__ == "__main__":
    # Inputs:
    # 1) .afa file.
    # 2) hmm database
    # 3) percent id
    args = create_parser()

    # if we can't find the .afa, exit
    if not os.path.isfile(args.aligned_fasta_file):
        raise ValueError(f"couldn't find .afa at {args.aligned_fasta_file}")

    esl_output = subprocess.check_output(
        f"esl-alistat {args.aligned_fasta_file}".split()
    ).decode("utf-8")
    esl_output = esl_output.split("\n")[2].split(":")[-1]

    if int(esl_output) < 10:
        log.info(f"less than 10 sequences found for {args.aligned_fasta_file}, exiting")
        exit()

    tblout_path = os.path.splitext(args.aligned_fasta_file)[0] + ".tblout"
    # make sure that the tblout passed in exists, otherwise error out
    if args.tblout is not None:
        if not os.path.isfile(args.tblout):
            raise ValueError(f"couldn't find .tblout at {args.tblout}")
        else:
            tblout_path = args.tblout
    # else, look for a tblout in the same directory as the .afa (with .tblout as the extension).
    # if it doesn't exist, make it.
    elif not os.path.isfile(tblout_path):
        subprocess.call(
            f"hmmsearch -o /dev/null --tblout {tblout_path} {args.hmmdb} {args.aligned_fasta_file}".split()
        )

    # cluster the .afa if we can't find the .ddgm
    ddgm_path = os.path.splitext(args.aligned_fasta_file)[0] + ".ddgm"
    if not os.path.isfile(ddgm_path):
        subprocess.call(f"carbs cluster {args.aligned_fasta_file}".split())

    carbs_output_template = (
        os.path.join(
            args.carbs_output_directory,
            os.path.splitext(os.path.basename(args.aligned_fasta_file))[0],
        )
        + ".{}-{}.fa"
    )
    fasta_output_template = (
        os.path.join(
            args.fasta_output_directory,
            os.path.splitext(os.path.basename(args.aligned_fasta_file))[0],
        )
        + ".{}-{}.fa"
    )

    os.makedirs(args.fasta_output_directory, exist_ok=True)
    os.makedirs(args.carbs_output_directory, exist_ok=True)

    # now, split the .afa at the given pid:
    if not os.path.isfile(carbs_output_template.format(args.pid, "train")):
        cmd = f"carbs split -T argument --split_test --output_path {args.carbs_output_directory} {args.aligned_fasta_file} {args.pid}"
        subprocess.call(cmd.split())

    # now, use the .tblout labels to create new fasta files with labels.
    tblout = parse_tblout(tblout_path)

    train_fasta_in = carbs_output_template.format(args.pid, "train")

    if os.path.isfile(train_fasta_in):
        train_fasta_out = fasta_output_template.format(args.pid, "train")
        labels_from_file(train_fasta_in, train_fasta_out, tblout)

    test_fasta_in = carbs_output_template.format(args.pid, "test")
    if os.path.isfile(test_fasta_in):
        test_fasta_out = fasta_output_template.format(args.pid, "test")
        labels_from_file(test_fasta_in, test_fasta_out, tblout)

    valid_fasta_in = carbs_output_template.format(args.pid, "valid")
    if os.path.isfile(valid_fasta_in):
        valid_fasta_out = fasta_output_template.format(args.pid, "valid")
        labels_from_file(valid_fasta_in, valid_fasta_out, tblout)
