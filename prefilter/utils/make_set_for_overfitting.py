import json
import os
import numpy as np

from argparse import ArgumentParser
from glob import glob
from collections import defaultdict
from random import shuffle

from utils import read_sequences_from_json

def _parser_setup():

    ap = ArgumentParser()
    ap.add_argument('--json-dir', required=True)
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--num-unique-classes', type=int, default=10)
    ap.add_argument('--num-samples-per-class', type=int, default=100)

    return ap.parse_args()

def _read_json(f):

    with open(f, 'r') as src:
        dct = json.load(src)
    return dct

def _invert_dict_of_lists(dct):
    
    out = defaultdict(list)
    for k in dct:
        if isinstance(dct[k], str):
            out[v].append(k)
        elif isinstance(dct[k], list):
            for v in dct[k]:
                out[v].append(k)
        else:
            print('this shouldn\'t happen')
    return out

def main(args):

    json_files = glob(os.path.join(args.json_dir, "*json"))
    sequence_to_label = {}

    for f in json_files:
        sequence_to_label.update(_read_json(f))

    label_to_sequence = _invert_dict_of_lists(sequence_to_label)
    label_to_count = {}
    for label, sequence_list in label_to_sequence.items():
        label_to_count[label] = len(sequence_list)

    labels = np.asarray(list(label_to_count.keys()))
    counts = np.asarray(list(label_to_count.values()))
    idx = np.argsort(counts)[::-1]
    labels = labels[idx]
    counts = counts[idx]

    i = 0
    data = defaultdict(list)
    while i < args.num_unique_classes: 

        if label_to_count[labels[i]] >= args.num_samples_per_class:
            sequences = label_to_sequence[labels[i]]
            data[labels[i]].extend(sequences[:args.num_samples_per_class])

        i += 1

    train = defaultdict(list)
    test = defaultdict(list)

    for label in data:
        sequences = data[label]
        shuffle(sequences)
        num_test = int(len(sequences)*0.2)

        train[label].extend(sequences[num_test:])
        test[label].extend(sequences[:num_test])

    train = _invert_dict_of_lists(train)
    test = _invert_dict_of_lists(test)

    train_fname = os.path.join(args.save_dir, 'train-subset.json')
    test_fname = os.path.join(args.save_dir, 'test-subset.json')

    with open(train_fname, 'w') as src:
        json.dump(train, src)
    with open(test_fname, 'w') as src:
        json.dump(test, src)

if __name__ == '__main__':
    # 
    args = _parser_setup()
    main(args)
