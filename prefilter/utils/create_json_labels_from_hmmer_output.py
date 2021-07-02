import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

def convert_hmmer_domtblout_to_json_labels(fname, single_best_score=False):
    '''ingests a hmmer domtblout file'''

    df = pd.read_csv(fname, skiprows=3, sep='\s+', engine='python')

    seq_name_to_family = defaultdict(list)

    if single_best_score:

        cnt = 0
        for _, row in df.iterrows():
            seq_name = row[0]
            if seq_name == '#':
                row = row.index
                seq_name = row[0]
            family = row[4]
            try:
                score = float(row[6])
            except (ValueError, TypeError):
                cnt += 1
                continue
            seq_name_to_family[seq_name].append((family, score))

        for seq_name in seq_name_to_family:
            maxidx = np.argmin([s[1] for s in seq_name_to_family[seq_name]])
            best_family = [s[0] for s in seq_name_to_family[seq_name]][maxidx]
            seq_name_to_family[seq_name] = [best_family]

        print('had to throw out {} rows'.format(cnt))

    else:

        for _, row in df.iterrows():
            seq_name = row[0]
            if seq_name == '#':
                row = row.index
                seq_name = row[0]
            family = row[4]
            seq_name_to_family[seq_name].append(family)

    return seq_name_to_family


def create_name_to_seq_dict(fasta_file):

    with open(fasta_file, 'r') as f:
        lines = f.read().split('>')[1:]

    name_to_seq = {}

    for line in lines:
        name = line[:line.find('\n')]
        seq = line[line.find('\n')+1:].rstrip('\n')
        seq = seq.replace('\n', '')
        name_to_seq[name] = seq

    return name_to_seq


def save_labels(seq_name_to_labels, 
                fasta_file_with_sequences,
                out_fname):

    sequence_name_to_sequence = create_name_to_seq_dict(fasta_file_with_sequences) # name to sequence
    tmp = {}
    # this takes care of different naming conventions b/t fasta files
    # without modifying the underlying files with sed or something
    for name in sequence_name_to_sequence.keys():
        x = name.find(' ')
        if x:
            new_name = name[:x]
            tmp[new_name] = sequence_name_to_sequence[name]

    sequence_name_to_sequence = tmp

    sequence_to_labels = {} # fasta sequence to label

    for seq_name, sequence in sequence_name_to_sequence.items():

        try:

            labels = seq_name_to_labels[seq_name]
            sequence_to_labels[sequence] = list(set(labels))

        except KeyError as e:
            print('keyerror', e, seq_name)

    with open(out_fname, 'w') as f:
        json.dump(sequence_to_labels, f, indent=2)

if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--domtblout',
                    type=str,
                    help='domtblout of hmmer',
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
    ap.add_argument('--single-best-score', 
                    action='store_true',
                    help='whether or not to save the single best score')


    args = ap.parse_args()
    if os.path.isfile(args.label_fname):
        print('already created {}, not creating new json\
                labels'.format(args.label_fname))
    else:
        sequences_and_labels = convert_hmmer_domtblout_to_json_labels(args.domtblout,
                args.single_best_score)

        save_labels(sequences_and_labels, 
                    args.sequences,
                    args.label_fname)
