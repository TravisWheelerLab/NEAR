import numpy as np
import os
import matplotlib.pyplot as plt
import json

from argparse import ArgumentParser
from glob import glob

def summarize_length_dist(json_files, savefig=None):

    lengths = []
    for f in json_files:
        with open(f, 'r') as src:
            sequence_to_labels = json.load(src)
        sequences = list(sequence_to_labels.keys())
        l = list(map(len, sequences))
        lengths += l

    s = 'mean length: {:.3f}, median: {:.3f}, 25th percentile: {:.3f}, 75th'
    s += ' percentile {:.3f}, number of seq. {:d}'

    s = s.format(np.mean(lengths),
            np.median(lengths),
            np.percentile(lengths, 25),
            np.percentile(lengths, 75),
            len(lengths))

    if savefig is not None:
        plt.figure()
        plt.title(s, fontsize=)
        plt.hist(lengths, bins=100)
        plt.xlabel('sequence length')
        plt.xlabel('number')
        plt.savefig(savefig)

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
    summarize_length_dist(json_files, savefig=save_path)
