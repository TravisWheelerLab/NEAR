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


def random_filename():
    return f"/tmp/{str(time.time())}"


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


def reclassify(fasta_file, hmm_file):

    if not os.path.isfile(fasta_file):
        log.info(f"{fasta_file} does not exist")
        return

    if not os.path.isfile(hmm_file):
        pfunc(f"{hmm_file} does not exist")
        return

    # reclassify the fasta file with the hmm:
    # wait. Do I want to reclassify the fasta file with all of the hmms from the training set?
    # probably... for now since we're doing single label classification this will do.

    tblout_path_random = random_filename()

    relabeled_path = os.path.splitext(fasta_file)[0] + "-relabeled.fa"
    subprocess.call(
        f"hmmsearch -o /dev/null --tblout {tblout_path_random} {hmm_file} {fasta_file}".split()
    )
    if os.path.isfile(tblout_path_random):
        tblout_df = parse_tblout(tblout_path_random)
        labels_from_file(fasta_file, relabeled_path, tblout_df, relabel=True)
    else:
        pfunc(
            f"couldn't find tblout {tblout_path_random}; did hmmsearch work correctly?"
        )

    os.remove(tblout_path_random)


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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-a", "--aligned_fasta_file", type=str)
    parser.add_argument(
        "-p",
        "--pid",
        type=float,
        help="percent identity to split the aligned fasta file",
        required=True,
    )
    parser.add_argument(
        "-hdb",
        "--hmmdb",
        default=None,
        help="hmm database to search the aligned fasta file against (if the --tblout file doesn't exist",
        required=True,
    )
    parser.add_argument(
        "-adb",
        "--alidb",
        default=None,
        help="database of alignments (ex; Pfam-A.seed)",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--clustered_output_directory",
        type=str,
        help="where to save the clustered .fa files.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fasta_output_directory",
        type=str,
        help="where to save the labeled fasta files",
        required=True,
    )
    parser.add_argument("--evalue_threshold", type=float, default=1e-5)
    return parser.parse_args()


def labels_from_file(
    fasta_in, fasta_out, tblout_df, evalue_threshold=1e-5, relabel=False
):
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
            if relabel:
                # if we're relabeling, don't add a delimiter
                fasta_header = f">{label} "
            else:
                # otherwise, add it in.
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
                    if relabel:
                        labelset.append("RL" + seq_label)
                    else:
                        labelset.append(seq_label)

            if len(labelset):
                fasta_header += " ".join(labelset) + "\n" + sequence + "\n"
                dst.write(fasta_header)


def main(args):
    """
    Creates labels for the prefilter task.
    I will try to describe the complicated process of generating labels.

    1) Ingest an aligned fasta file (.afa) and use carbs (https://github.com/TravisWheelerLab/carbs) to split it into train,
    test, and validation sets. If there are < 3 sequences in the .afa, exit. If the .afa can't be split at the given
    percent identity, dump all of the sequences in it into the training set.
    2) Use the provided hmm database to classify each of the sequences in the .afa file. A sequence is considered
    "classified" if the alignment between it and a hmm produces an e-value of < 1e-5. The threshold is tunable via
    the --evalue command line argument. Each sequence can have multiple labels. Labels are provided in the form
    of a Pfam accession ID.  All of the sequences in the .afa are assumed to be present in the hmm database. The hmm
    database is used to assign "gold-standard" labels - i.e. the "true" family membership of each sequence in the
    .afa.
    3) Extract the sub-alignment of the training sequences from the alignment database and use the sub-alignment to
    build a new "train" hmm.
    4) Use the train hmm to reclassify the train, test, and validation splits, and save the sequences to a new file,
    called "{fasta_file}-relabeled.fa." The -relabeled files describe how well hmmer does when classifying the test/valid
    sequences when all it's seen is the training set. The "relabels" will be used to benchmark the neural network
    model that's aimed at prefilter hmmer hits. In effect, the "relabels" tell us how well our network actually matches
    hmmer - if hmmer can't classify a sequence given the training set, and we pass it that hard-to-classify sequence
    with our nn, we won't actually classify it correctly with hmmer. That's why we have two sets of labels - to quantify
    how well do we do at determining the "true" family membership, and how well we do compared to hmmer.

    Note: I use "classify" as shorthand for "align each hmm to each sequence in the fasta database with hmmsearch".

    :param args:
    :type args:
    :return:
    :rtype:
    """

    # if we can't find the .afa, exit
    if not os.path.isfile(args.aligned_fasta_file):
        raise ValueError(f"couldn't find .afa at {args.aligned_fasta_file}")

    esl_output = subprocess.check_output(
        f"esl-alistat {args.aligned_fasta_file}".split()
    ).decode("utf-8")
    esl_output = esl_output.split("\n")[2].split(":")[-1]

    if int(esl_output) < 3:
        pfunc(f"less than 3 sequences found for {args.aligned_fasta_file}, exiting")
        exit()

    tblout_path = os.path.splitext(args.aligned_fasta_file)[0] + ".tblout"
    if not os.path.isfile(tblout_path) or os.stat(tblout_path).st_size == 0:
        subprocess.call(
            f"hmmsearch -o /dev/null --tblout {tblout_path} {args.hmmdb} {args.aligned_fasta_file}".split()
        )

    # cluster the .afa if we can't find the .ddgm
    ddgm_path = os.path.splitext(args.aligned_fasta_file)[0] + ".ddgm"

    if not os.path.isfile(ddgm_path) or os.stat(ddgm_path).st_size == 0:
        subprocess.call(f"carbs cluster {args.aligned_fasta_file}".split())

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

    os.makedirs(args.fasta_output_directory, exist_ok=True)
    os.makedirs(args.clustered_output_directory, exist_ok=True)

    # split the .afa at the given pid:
    cmd = f"carbs split -T argument --split_test --output_path {args.clustered_output_directory} {args.aligned_fasta_file} {args.pid}"
    subprocess.call(cmd.split())

    # use the .tblout labels to create new fasta files with labels.

    # these are called "true" labels, as the hmmdb contains all of the Pfam-A seed alignments.
    tblout = parse_tblout(tblout_path)

    train_fasta_in = clustered_output_template.format(args.pid, "train")

    if os.path.isfile(train_fasta_in):
        train_fasta_out = fasta_output_template.format(args.pid, "train")
        labels_from_file(train_fasta_in, train_fasta_out, tblout)
    else:
        raise ValueError(f"train file from {train_fasta_in} was not created by carbs.")

    test_fasta_in = clustered_output_template.format(args.pid, "test")
    test_fasta_out = None
    if os.path.isfile(test_fasta_in):
        test_fasta_out = fasta_output_template.format(args.pid, "test")
        labels_from_file(test_fasta_in, test_fasta_out, tblout)

    valid_fasta_in = clustered_output_template.format(args.pid, "valid")
    valid_fasta_out = None
    if os.path.isfile(valid_fasta_in):
        valid_fasta_out = fasta_output_template.format(args.pid, "valid")
        labels_from_file(valid_fasta_in, valid_fasta_out, tblout)

    # grab the sequences from the alignment that are in train
    train_seq, _ = utils.fasta_from_file(train_fasta_out)
    if not len(train_seq):
        pfunc(
            f"No training sequences in {train_fasta_out}."
            f" Check {train_fasta_in} and {tblout_path}."
        )
        return

    random_f = random_filename()
    with open(random_f, "w") as dst:
        for seq in train_seq:
            delim = seq.find(" |")
            if delim != -1:
                seq_name = seq[:delim].split()[0]
                dst.write(f"{seq_name}\n")
            else:
                raise ValueError(
                    f"has {train_fasta_out} been relabeled with " f"label_fasta.py?"
                )

    # grab the training alignment
    ali_out_path = random_filename()

    # then grab the families' alignment (after first reformatting to stockholm)
    # this won't propagate the name of the alignment. So rely on the alidb to
    # grab the training alignment. This assumes that the aligned fasta file's
    # name describes the family from which the sequences come from...

    stockholm_out_path = os.path.splitext(args.aligned_fasta_file)[0] + ".sto"
    if (
        not os.path.isfile(stockholm_out_path)
        or os.stat(stockholm_out_path).st_size == 0
    ):
        family_name = os.path.splitext(os.path.basename(args.aligned_fasta_file))[0]
        assert (
            subprocess.call(
                f"esl-afetch -o {stockholm_out_path} {args.alidb} {family_name}".split()
            )
            == 0
        )

    print(random_f, stockholm_out_path, ali_out_path)

    assert (
        subprocess.call(
            f"esl-alimanip -o {ali_out_path} --seq-k {random_f} {stockholm_out_path}".split()
        )
        == 0
    )

    os.remove(random_f)

    hmm_out_path = random_filename()
    # use the train alignment to build a new hmm
    subprocess.call(f"hmmbuild -o /dev/null {hmm_out_path} {ali_out_path}".split())
    # classify our train, test, and valid sequences with the training hmm
    reclassify(train_fasta_out, hmm_out_path)

    if test_fasta_out is not None:
        reclassify(test_fasta_out, hmm_out_path)
    if valid_fasta_out is not None:
        reclassify(valid_fasta_out, hmm_out_path)

    os.remove(hmm_out_path)

    n_train = subprocess.check_output(f"esl-seqstat {train_fasta_out}".split()).decode(
        "utf-8"
    )
    n_train = int(n_train.split("\n")[2].split(":")[-1])
    if n_train == 0:
        raise ValueError(f"{train_fasta_out} empty")


if __name__ == "__main__":
    main(parse_args())
