import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

def convert_hmmer_domtblout_to_json_labels(fname,  fout):
    '''ingets a hmmer domtblout file'''
    df = pd.read_csv(fname, skiprows=3,sep='\s+', engine='python')

    seq_name_to_family = defaultdict(list)

    for _, row in df.iterrows():
        seq_name = row[0]
        family = row[4]
        seq_name_to_family[seq_name].append(family)

    # l = list(map(len, list(seq_name_to_family.values())))

    with open(fout, 'w') as f:
        json.dump(seq_name_to_family, f)

    return seq_name_to_family


def create_name_to_seq_dict(fasta_file):
    with open(fasta_file, 'r') as f:
        lines = f.read().split('>')[1:]
    name_to_seq = {}
    for line in lines:
        name = line[:line.find('\n')]
        seq = line[line.find('\n')+1:].rstrip('\n')
        name_to_seq[name] = seq
    return name_to_seq


def save_labels(json_file_with_name_to_labels, fasta_file_with_sequences,
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

    with open(json_file_with_name_to_labels, 'r') as f:
        seq_to_labels = json.load(f) # sequence name to label

    fstl = {} # fasta sequence to label
    for seq, labels in seq_to_labels.items():
        try:
            fstl[nts[seq]] = labels
        except KeyError as e:
            print(e)

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

    fout = os.path.splitext(os.path.basename(args.domtblout))[0] + '_labels.json'

    # use hmmer output to map sequence names to family
    if not os.path.isfile(fout):
        convert_hmmer_domtblout_to_json_labels(args.domtblout, fout)
    else:
        print("domtblout {} already converted to json {}".format(args.domtblout,
            fout))


    # subprocess.popen?
    # name_to_sequence = create_name_to_seq_dict(fasta_from_esl_sfetch)
    save_labels(fout, 
                args.sequences,
                args.label_fname)
