import numpy as np
import os
import matplotlib.pyplot as plt
import json

from argparse import ArgumentParser
from collections import defaultdict
from glob import glob

def summarize_dataset(json_files, savefig=None):

    lengths = []
    sequence_to_count = defaultdict(int)
    labels_per_sequence = []

    for f in json_files:
        with open(f, 'r') as src:
            sequence_to_labels = json.load(src)
        sequences = list(sequence_to_labels.keys())
        labels = list(sequence_to_labels.values())
        l = list(map(len, sequences))
        lengths += l
        l = list(map(len, labels))
        labels_per_sequence += l

    s = 'mean length: {:.3f}, median: {:.3f}, 25th percentile: {:.3f}, 75th'
    s += ' percentile {:.3f}, number of seq. {:d}'

    s = s.format(np.mean(lengths),
            np.median(lengths),
            np.percentile(lengths, 25),
            np.percentile(lengths, 75),
            len(lengths))

    if savefig is not None:
        fig, ax = plt.subplots(ncols=2)

        ax[0].set_title(s, fontsize=8)
        ax[0].hist(lengths, bins=100, histtype='step')
        ax[0].set_xlabel('sequence length')
        ax[0].set_ylabel('number')

        ax[1].set_title('number of labels per sequence')
        ax[1].hist(labels_per_sequence, bins=100, histtype='step')
        ax[1].set_xlabel('num labels')
        ax[1].set_ylabel('number')
        
        plt.savefig(savefig)
        plt.close()

def parser():

    ap = ArgumentParser()
    ap.add_argument('--directory', required=True, type=str)
    ap.add_argument('--glob_string', required=True, type=str)
    ap.add_argument('--save_fig', type=str)
    return ap.parse_args()


if __name__ == '__main__':

    args = parser()

    directory = args.directory
    glob_string = args.glob_string
    save_path = args.save_fig
    json_files = glob(os.path.join(directory, glob_string))
    summarize_dataset(json_files, savefig=save_path)

