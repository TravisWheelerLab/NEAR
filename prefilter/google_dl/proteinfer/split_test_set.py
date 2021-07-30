import json
import sys
import os
import numpy as np

from glob import glob
from argparse import ArgumentParser

def parser():

    ap = ArgumentParser()
    ap.add_argument('--directory', required=True,
            help='directory containing test files')
    ap.add_argument('--glob_str', required=False,
            help='glob string to pick out test files',
            default='*test.json')
    args = ap.parse_args()
    return args

def split_files(test_files, args):

    '''
    Splits a json file containing sequences and their hmmer labels into two
    files, each containing half of the sequences as in the original file (50/50
    split). Names one half <original-file>valid-split.json and the other half
    <origina-file>test-split.json.
    '''

    suffix = args.glob_str
    suffix = suffix.replace('*', '')

    for f in test_files:

        valid_filename = f.replace(suffix, 'valid-split.json')
        test_filename = f.replace(suffix, 'test-split.json')
        with open(f, 'r') as src:
            sequence_to_label = json.load(src)

        sequences = list(sequence_to_label.keys())

        if len(sequences) > 1:
            valid = np.random.choice(sequences,
                                     size=int(len(sequences)*0.5), 
                                     replace=False)
            
            valid_sequence_to_label = {}
            for v in valid:
                valid_sequence_to_label[v] = sequence_to_label[v]
                del sequence_to_label[v]
            with open(valid_filename, 'w') as dst:
                json.dump(valid_sequence_to_label, dst)

            with open(test_filename, 'w') as dst:
                json.dump(sequence_to_label, dst)


        else:
            print('only 1 seq in test, not splitting into valid (but saving nonetheless!)')
            with open(test_filename, 'w') as dst:
                json.dump(sequence_to_label, dst)


def main(args):

    test_files = glob(os.path.join(args.directory, args.glob_str))
    split_files(test_files, args)

if __name__ == '__main__':

    args = parser()
    main(args)

