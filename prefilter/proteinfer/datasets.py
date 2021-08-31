import json
import os
import time
from glob import glob

import torch

import inference

__all__ = ['Word2VecStyleDataset',
           'ProteinSequenceDataset',
           'SimpleSequenceEmbedder'
           ]

inferrer = inference.Inferrer(
    '/home/tc229954/data/prefilter/proteinfer/trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760',
    use_tqdm=False,
    batch_size=1,
    activation_type="pooled_representation"
)


class ProteinSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, json_files,
                 existing_name_to_label_mapping=None,
                 evaluating=False):

        self.json_files = json_files
        self.existing_name_to_label_mapping = existing_name_to_label_mapping
        self.evaluating = evaluating

        self._build_dataset()

    def _encoding_func(self, x):

        return inferrer.get_activations([x.upper()])

    def _build_dataset(self):

        self.sequences_and_labels = []

        if self.existing_name_to_label_mapping is None:
            self.name_to_class_code = {}
            class_id = 0

        elif isinstance(self.existing_name_to_label_mapping, str):
            s = 'loading class mapping from file {}'.format(self.existing_name_to_label_mapping)
            print(s)

            with open(self.existing_name_to_label_mapping, 'r') as src:
                self.name_to_class_code = json.load(src)

            class_id = len(self.name_to_class_code)

        elif isinstance(self.existing_name_to_label_mapping, dict):
            self.name_to_class_code = self.existing_name_to_label_mapping
            class_id = len(self.name_to_class_code)

        else:
            s = 'expected existing_name_to_label_mapping to be one of dict, string, or None, found {}'.format(type(self.existing_name_to_label_mapping))
            raise ValueError(s)



        for j in self.json_files:

            with open(j, 'r') as src:
                sequence_to_labels = json.load(src)

            for sequence, labelset in sequence_to_labels.items():

                for label in labelset:

                    if label not in self.name_to_class_code:

                        if not self.evaluating:
                            self.name_to_class_code[label] = class_id
                            class_id += 1

                self.sequences_and_labels.append([labelset, sequence])

        if self.existing_name_to_label_mapping is not None and not self.evaluating:
            print('saving to existing name to label_mapping')

            with open(self.existing_name_to_label_mapping, 'w') as dst:
                json.dump(self.name_to_class_code, dst)
        else:

            self.class_code_mapping = './pfam_names_to_class_id-{}.json'.format(time.time())

            with open(self.class_code_mapping, 'w') as dst:
                json.dump(self.name_to_class_code, dst)

        self.n_classes = class_id + 1

    def __len__(self):

        return len(self.sequences_and_labels)

    def _make_multi_hot(self, labels):
        y = torch.zeros(self.n_classes)
        class_ids = [self.name_to_class_code[l] for l in labels]
        for idx in class_ids:
            y[idx] = 1
        return y

    def __getitem__(self, idx):

        labels, features = self.sequences_and_labels[idx][0], self.sequences_and_labels[idx][1]
        x = self._encoding_func(features)
        y = self._make_multi_hot(labels)
        return torch.tensor(x), y


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


class SimpleSequenceEmbedder(torch.utils.data.Dataset):

    def __init__(self, fasta_file):
        """
        takes a fasta file as input
        """

        self.fasta_file = fasta_file
        self.labels, self.sequences = fasta_from_file(fasta_file)

    @staticmethod
    def _encoding_func(x):
        return inferrer.get_activations([x.upper()])

    def __getitem__(self, idx):

        return torch.tensor(self._encoding_func(self.sequences[idx]))

    def __len__(self):
        return len(self.sequences)


if __name__ == '__main__':

    sse = SimpleSequenceEmbedder('/home/tc229954/data/prefilter/small-dataset/random_sequences/random_sequences.fa')
    print(len(sse))
    for s in sse:
        print(s.shape)
