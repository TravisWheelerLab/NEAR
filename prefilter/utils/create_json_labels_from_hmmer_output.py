import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

def convert_hmmer_domtblout_to_json_labels(fname):
    '''ingets a hmmer domtblout file'''
    df = pd.read_csv(fname, skiprows=3,sep='\s+', engine='python')

    seq_name_to_family = defaultdict(list)

    for _, row in df.iterrows():
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


def save_labels(seq_to_labels, fasta_file_with_sequences,
        out_fname):

    nts = create_name_to_seq_dict(fasta_file_with_sequences) # name to sequence
    nts_ = {}
    # this takes care of different naming conventions b/t fasta files
    # without modifying the underlying files with sed or something
    for name in nts.keys():
        x = name.find(' ')
        if x:
            new_name = name[:x]
            nts_[new_name] = nts[name]

    nts = nts_

    fstl = {} # fasta sequence to label
    for seq, labels in seq_to_labels.items():
        try:
            fstl[nts[seq]] = labels
        except KeyError as e:
            print('keyerror', e)

    with open(out_fname, 'w') as f:
        json.dump(fstl, f)

if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--domtblout', type=str,
                    help='domtblout of hmmer')
    ap.add_argument('--sequences', type=str, 
                    help='fasta file containing sequences')
    ap.add_argument('--label-fname', type=str,
            help='where to save the json file mapping fasta sequence names to labels') 

    args = ap.parse_args()

    sequences_and_labels = convert_hmmer_domtblout_to_json_labels(args.domtblout)

    save_labels(sequences_and_labels, 
                args.sequences,
                args.label_fname)
