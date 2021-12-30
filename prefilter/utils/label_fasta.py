#!/usr/bin/env python3
import pdb
import subprocess
import pandas as pd
import logging
import os
import time

log = logging.getLogger(__name__)

from argparse import ArgumentParser
import prefilter.utils as utils

# TODO: replace hardcoded stuff with variables in __init__.py in prefilter


def random_filename(directory="/tmp/"):
    """
    Generate a random filename for temporary use.

    :param directory: Which directory to place the file in
    :type directory: str
    :return: random filename in directory
    :rtype: str
    """
    return os.path.join(directory, str(time.time()))


def pfunc(str):
    print(str)


TBLOUT_COL_NAMES = [
    "target_name",
    "query_name",
    "accession_id",
    "e_value",
    "description",
]
TBLOUT_COLS = [0, 2, 3, 4, 18]


def parse_tblout(tbl):
    """
    Parse a .tblout file created with hmmsearch -o <tbl>.tblout <seqdb> <hmmdb>
    :param tbl: .tblout filename.
    :type tbl: str
    :return: dataframe containing the rows of the .tblout.
    :rtype: pd.DataFrame
    """
    if os.path.splitext(tbl)[1] != ".tblout":
        raise ValueError(f"must pass a .tblout file, found {tbl}")

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


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="action", dest="command")

    split_parser = subparsers.add_parser(name="split", add_help=False)
    split_parser.add_argument("-a", "--aligned_fasta_file", type=str)
    split_parser.add_argument(
        "-db",
        "--hmmdb",
        default=None,
        help="hmm database to search the aligned fasta file against (if the --tblout file doesn't exist",
        required=True,
    )
    split_parser.add_argument(
        "-p",
        "--pid",
        type=float,
        help="percent identity to split the aligned fasta file",
        required=True,
    )
    split_parser.add_argument(
        "-adb",
        "--alidb",
        default=None,
        help="database of alignments (ex; Pfam-A.seed)",
        required=True,
    )
    split_parser.add_argument(
        "-c",
        "--clustered_output_directory",
        type=str,
        help="where to save the clustered .fa files.",
        required=True,
    )
    split_parser.add_argument(
        "-o",
        "--fasta_output_directory",
        type=str,
        help="where to save the labeled fasta files",
        required=True,
    )

    split_parser.add_argument("--evalue_threshold", type=float, default=1e-5)

    label_parser = subparsers.add_parser("label")

    label_parser.add_argument("fasta_file", help="fasta file to label")
    label_parser.add_argument(
        "hmmdb",
        default=None,
        help="hmm database to search the aligned fasta file against",
    )
    label_parser.add_argument(
        "-o",
        "--fasta_output_directory",
        help="where to save the labeled fasta files",
        required=True,
    )

    train_hdb_parser = subparsers.add_parser("hdb")
    train_hdb_parser.add_argument(
        "fasta_file", help="fasta file containing train sequences"
    )
    train_hdb_parser.add_argument("alidb", help="alignment database")
    train_hdb_parser.add_argument("-o", "--overwrite", action="store_true")
    return parser


def labels_from_file(fasta_in, fasta_out, tblout_df, evalue_threshold=1e-5):
    if os.path.isfile(fasta_out):
        pfunc(f"Already created labels for {fasta_out}.")
        return

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
                # or the hmmdb can't find them.
                pfunc(
                    f"sequence named {target_label} not found in classification on {fasta_in}"
                )
                target_label = target_label.split()[0]
                # Sometimes the names don't have a description, and aren't found.
                assigned_labels = tblout_df.loc[
                    tblout_df["target_name"] == target_label
                ]
                if len(assigned_labels) == 0:
                    pfunc(
                        f"sequence named {target_label} not found in classification on {fasta_in}"
                    )
                continue
            # each sequence should have at least one label, but we
            # only need to grab one since one sequence can be associated with
            # multiple pfam accession IDs
            fasta_header = f">{label} | "

            labelset = []

            for seq_label, e_value in zip(
                assigned_labels["accession_id"], assigned_labels["e_value"]
            ):
                if "PF" not in seq_label:
                    raise ValueError(
                        f"Pfam accession ID not found in labels in {tblout_df}"
                    )

                if float(e_value) <= evalue_threshold:
                    labelset.append(seq_label)

            if len(labelset):
                fasta_header += " ".join(labelset) + "\n" + sequence + "\n"
                dst.write(fasta_header)


def cluster_and_split_sequences(
    aligned_fasta_file, clustered_output_directory, pid, overwrite=False
):

    output_template = (
        os.path.join(
            clustered_output_directory,
            os.path.splitext(os.path.basename(aligned_fasta_file))[0],
        )
        + ".{}-{}.fa"
    )

    os.makedirs(clustered_output_directory, exist_ok=True)

    ddgm_path = os.path.splitext(aligned_fasta_file)[0] + ".ddgm"
    # cluster the .afa if we can't find the .ddgm
    if not os.path.isfile(ddgm_path) or os.stat(ddgm_path).st_size == 0:
        subprocess.call(f"carbs cluster {aligned_fasta_file}".split())

    if not os.path.isfile(output_template.format(pid, "train")):
        cmd = f"carbs split -T argument --split_test --output_path {clustered_output_directory} {aligned_fasta_file} {pid}"
        subprocess.call(cmd.split())
    else:
        pfunc(f"already created {output_template.format(pid, 'train')}")


def label_with_hmmdb(fasta_file, fasta_outfile, hmmdb):

    tblout_path = os.path.splitext(fasta_file)[0] + ".tblout"

    if not os.path.isfile(tblout_path) or os.stat(tblout_path).st_size == 0:
        subprocess.call(
            f"hmmsearch -o /dev/null --tblout {tblout_path} {hmmdb} {fasta_file}".split()
        )

    tblout = parse_tblout(tblout_path)

    labels_from_file(fasta_file, fasta_outfile, tblout)


def extract_ali_and_create_hmm(fasta_file, alidb, overwrite=False):

    if "train" not in os.path.basename(fasta_file):
        raise ValueError(f'{fasta_file} does not have "train" in it.')

    stockholm_out = os.path.splitext(fasta_file)[0] + ".sto"

    if not os.path.isfile(stockholm_out) or overwrite:
        # extract the alignment from the alidb that has the same
        # name as the fasta file
        family_name = os.path.basename(fasta_file)
        family_name = family_name[: family_name.find(".")]
        train_alignment_temp_file = random_filename()
        cmd = f"esl-afetch -o {train_alignment_temp_file} {alidb} {family_name}"
        assert subprocess.call(cmd.split()) == 0

        train_name_file = random_filename()
        train_names, _ = utils.fasta_from_file(fasta_file)

        with open(train_name_file, "w") as dst:
            for name in train_names:
                dst.write(f"{name.split()[0]}\n")

        pfunc(len(train_names))
        if len(train_names):
            cmd = f"esl-alimanip -o {stockholm_out} --seq-k {train_name_file} {train_alignment_temp_file}"
            # check for successful exit code
            assert subprocess.call(cmd.split()) == 0
        os.remove(train_name_file)
    else:
        pfunc(f"{stockholm_out} already exists.")

    hmm_out_path = os.path.splitext(fasta_file)[0] + ".hmm"

    if not os.path.isfile(hmm_out_path) or overwrite:
        # create the hmm
        cmd = f"hmmbuild -o /dev/null -n 1 {hmm_out_path} {train_alignment_temp_file}"
        assert subprocess.call(cmd.split()) == 0
    else:
        pfunc(f"{hmm_out_path} already exists.")

    os.remove(train_alignment_temp_file)


def split_and_label_sequences(args):
    """
    Creates labels for the prefilter task.
    :param args:
    :type args:
    :return:
    :rtype:
    """

    # if we can't find the .afa, exit
    if not os.path.isfile(args.aligned_fasta_file):
        raise ValueError(f"couldn't find .afa at {args.aligned_fasta_file}")

    # use esl-alistat to get the number of sequences in the fasta file
    esl_output = subprocess.check_output(
        f"esl-alistat {args.aligned_fasta_file}".split()
    ).decode("utf-8")
    esl_output = esl_output.split("\n")[2].split(":")[-1]

    # exit if there are less than 3 sequences
    if int(esl_output) < 3:
        pfunc(f"less than 3 sequences found for {args.aligned_fasta_file}, exiting")
        exit()

    # determine if there's a tblout (signaling that we've already classified these sequences with hmmsearch)

    os.makedirs(args.fasta_output_directory, exist_ok=True)

    # split the .afa at the given pid:

    clustered_output_template = (
        os.path.join(
            args.clustered_output_directory,
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

    cluster_and_split_sequences(
        args.aligned_fasta_file, args.clustered_output_directory, args.pid
    )

    train_fasta_in = clustered_output_template.format(args.pid, "train")
    train_fasta_out = fasta_output_template.format(args.pid, "train")

    label_with_hmmdb(train_fasta_in, train_fasta_out, args.hmmdb)

    test_fasta_in = clustered_output_template.format(args.pid, "test")
    valid_fasta_in = clustered_output_template.format(args.pid, "valid")

    if os.path.isfile(test_fasta_in):
        test_fasta_out = fasta_output_template.format(args.pid, "test")
        label_with_hmmdb(test_fasta_in, test_fasta_out, args.hmmdb)

    if os.path.isfile(valid_fasta_in):
        valid_fasta_out = fasta_output_template.format(args.pid, "valid")
        label_with_hmmdb(valid_fasta_in, valid_fasta_out, args.hmmdb)

    # grab the sequences from the alignment that are in train
    train_seq, _ = utils.fasta_from_file(train_fasta_out)

    if not len(train_seq):
        pfunc(
            f"No training sequences in {train_fasta_out}."
            f" Check {train_fasta_in} and {tblout_path}."
        )
        return


if __name__ == "__main__":
    program_parser = parse_args()
    program_args = program_parser.parse_args()
    if program_args.command == "split":
        split_and_label_sequences(program_args)
    elif program_args.command == "label":
        fasta_outf = os.path.join(
            program_args.fasta_output_directory,
            os.path.basename(program_args.fasta_file),
        )
        os.makedirs(program_args.fasta_output_directory, exist_ok=True)
        label_with_hmmdb(program_args.fasta_file, fasta_outf, program_args.hmmdb)
    elif program_args.command == "hdb":
        extract_ali_and_create_hmm(
            program_args.fasta_file, program_args.alidb, program_args.overwrite
        )
    else:
        pfunc(program_args)
        program_parser.print_help()
