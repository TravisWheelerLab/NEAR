#!/usr/bin/env python3
import sys
from collections import defaultdict

from argparse import ArgumentParser, FileType


def fasta_from_file(fasta_file):
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []

    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        sequence_strs.append("".join(buf))
        cur_seq_label = None
        buf = []

    with open(fasta_file, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:  # sequence line
                buf.append(line.strip())

    _flush_current_seq()

    return sequence_labels, sequence_strs


def _name_to_label(domtblout):
    name_to_label = defaultdict(list)
    for row in domtblout:
        row = row.rstrip('\n').split(' ')
        name_to_label[row[0]].append([row[2], row[3]])
    return name_to_label


def main(args):
    # names are the names as labeled in pfam.
    # we're going to overwrite these with new "names" that are actually
    # labels - the pfam accession ID.
    names, seq = fasta_from_file(args.fasta_file)
    names = [n.split(' ')[0] for n in names]
    name_to_seq = {n: s for n, s in zip(names, seq)}
    name_to_label = _name_to_label(args.domtblout_labels)
    with open(args.fasta_file, 'w') as dst:
        for sequence_name, labels in name_to_label.items():
            labels = sorted(labels, key=lambda x: x[1])
            labels = list(filter(lambda x: float(x[1]) <= args.evalue_threshold, labels))
            # sort so the first label will be the one with the highest e-value
            try:
                seq = name_to_seq[sequence_name]
            except KeyError:
                print('a key error means that the domtblout didn\'t contain anything for the sequence named {}'
                      'in {}'.format(sequence_name, args.fasta_file))
                continue
            labels = [l[0] for l in labels]
            dst.write(
                ">" + sequence_name + ' | ' + ' '.join(labels) + '\n' + seq
            )
            dst.write('\n')


def parser():
    parser = ArgumentParser()
    parser.add_argument('--fasta_file', type=str)
    parser.add_argument('domtblout_labels', nargs='?', type=FileType('r'),
                        default=sys.stdin, help='')
    parser.add_argument('--evalue_threshold', type=float, default=1e-5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    main(args)
