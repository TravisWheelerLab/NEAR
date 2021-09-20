#!/usr/bin/env python3
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd


# W9XJ26_9EURO/7-77

def convert_hmmer_domtblout_to_json_labels(fname, single_best_score=False,
                                           evalue_threshold=None):
    """ingests a hmmer domtblout file"""

    df = pd.read_csv(fname, skiprows=2, sep="\s+", engine="python")
    names = df.iloc[:, 0]
    families = df.iloc[:, 4]
    e_values = df.iloc[:, 6]
    not_bad = names != '#'
    names = names[not_bad]
    families = families[not_bad]
    e_values = e_values[not_bad]
    sequence_name_to_family = defaultdict(list)

    if single_best_score:
        min_evalue = np.inf
        min_family_name = None
        min_sequence_name = None
        for name, family, e_value in zip(names, families, e_values):
            if float(e_value) < min_evalue:
                min_family_name = family
                min_evalue = e_value
                min_sequence_name = name
        sequence_name_to_family[min_sequence_name].append(min_family_name)
    else:
        for name, family, e_value in zip(names, families, e_values):
            if float(e_value) < evalue_threshold:
                sequence_name_to_family[name].append(family)

    return sequence_name_to_family


def create_name_to_seq_dict(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.read().split('>')[1:]

    name_to_seq = {}

    for line in lines:
        name = line[:line.find('\n')]
        seq = line[line.find('\n') + 1:].rstrip('\n')
        seq = seq.replace('\n', '')
        name_to_seq[name] = seq

    return name_to_seq


def save_labels(seq_name_to_labels,
                fasta_file_with_sequences,
                out_fname):
    sequence_name_to_sequence = create_name_to_seq_dict(fasta_file_with_sequences)  # name to sequence
    tmp = {}
    # this takes care of different naming conventions b/t fasta files
    # without modifying the underlying files with sed or something
    for name in sequence_name_to_sequence.keys():
        x = name.find(' ')
        if x:
            new_name = name[:x]
            tmp[new_name] = sequence_name_to_sequence[name]

    sequence_name_to_sequence = tmp

    sequence_to_labels = {}  # fasta sequence to label

    for seq_name, sequence in sequence_name_to_sequence.items():

        try:

            labels = list(set(seq_name_to_labels[seq_name]))
            if not len(labels):
                print('didn\'t find any labels for {}, {}'.format(seq_name,
                                                                  os.path.basename(fasta_file_with_sequences)))
            else:
                sequence_to_labels[sequence] = list(set(labels))

        except KeyError as e:
            raise KeyError("this really shouldn't happen, fix it.")

    if len(sequence_to_labels):
        with open(out_fname, 'w') as f:
            json.dump(sequence_to_labels, f, indent=2)
    else:
        print("couldn't find any labels over set evalue threshold for {}, not saving.".format(fasta_file_with_sequences))


def parser():
    ap = ArgumentParser()
    ap.add_argument('--domtblout',
                    type=str,
                    help='domtblout produced by hmmer',
                    required=True)

    ap.add_argument('--sequences',
                    type=str,
                    help='fasta file containing sequences',
                    required=True)

    ap.add_argument('--label-fname',
                    type=str,
                    help='where to save the json file mapping fasta sequence\
                    names to\
                    labels',
                    required=True)

    ap.add_argument('--overwrite',
                    action='store_true',
                    help='overwrite json files?')

    ap.add_argument('--single-best-score',
                    action='store_true',
                    help='whether or not to save the single best score')

    ap.add_argument('--evalue-threshold',
                    type=float,
                    default=1e-5,
                    help='overwrite json files?')

    parser_args = ap.parse_args()

    return parser_args


def main(args):
    if os.path.isfile(args.label_fname) and not args.overwrite:
        print('already created {}, not creating new json\
                labels'.format(args.label_fname))
    else:

        sequences_and_labels = convert_hmmer_domtblout_to_json_labels(args.domtblout,
                                                                      args.single_best_score,
                                                                      args.evalue_threshold)
        if sequences_and_labels is not None:
            save_labels(sequences_and_labels,
                        args.sequences,
                        args.label_fname)

        save_labels(sequences_and_labels,
                    args.sequences,
                    args.label_fname)


if __name__ == '__main__':
    args = parser()
    main(args)
