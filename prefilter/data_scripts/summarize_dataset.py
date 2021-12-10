import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from prefilter.utils import fasta_from_file, parse_labels


def histogram_number_of_seqs_per_family(fasta_files, savefig=None):
    """
    Creates and saves a histogram of the number of sequences in the list of fasta files.
    Useful for investigating the imbalance present in a dataset.
    :param savefig: where to save the data
    :type savefig: str
    :param fasta_files: files to anaylze
    :type fasta_files: str
    :return: None
    :rtype: None
    """
    label_to_sequence_count = defaultdict(int)
    for fasta in fasta_files:
        labelset, sequences = fasta_from_file(fasta)
        # parse labels, get
        for labelstring, sequence in zip(labelset, sequences):
            labels = parse_labels(labelstring)
            if labels is None:
                print(labelstring)
                continue
            else:
                for pfam_accession_id in labels:
                    label_to_sequence_count[pfam_accession_id] += 1

    fig, ax = plt.subplots()
    for i, (label, count) in enumerate(label_to_sequence_count.items()):
        ax.bar(i, count)
    ax.set_xticklabels(list(label_to_sequence_count.keys()))

    plt.savefig(savefig)
    plt.close()

def parser():
    ap = ArgumentParser()
    ap.add_argument(
        "--directory", required=True, type=str, help="where json files are stored"
    )
    ap.add_argument(
        "--glob_string", default="*fa", type=str, help="which pattern you want to match"
    )
    ap.add_argument("--save_fig", type=str, default="test.png")
    return ap.parse_args()


if __name__ == "__main__":
    args = parser()

    directory = args.directory
    glob_string = args.glob_string
    save_path = args.save_fig
    fasta_files = glob(os.path.join(directory, glob_string))
    if not len(fasta_files):
        raise ValueError(
            "glob string {}j does not contain any fasta files".format(
                os.path.join(directory, glob_string)
            )
        )
    histogram_number_of_seqs_per_family(fasta_files, savefig=save_path)
