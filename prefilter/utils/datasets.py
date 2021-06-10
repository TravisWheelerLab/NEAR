import os
import json
import torch
import numpy as np

import pdb
import time

from glob import glob
from random import shuffle
from collections import defaultdict

from . import utils as utils

__all__ = ['Word2VecStyleDataset',
           'ProteinSequenceDataset',
           ]


class Word2VecStyleDataset(torch.utils.data.Dataset):
    
    def __init__(self,
            json_files,
            max_sequence_length,
            name_to_label_mapping,
            n_negative_samples=5,
            evaluating=False
            ):

        self.max_sequence_length = max_sequence_length
        self.name_to_label_mapping = name_to_label_mapping
        self.n_negative_samples = n_negative_samples
        self.evaluating = evaluating
        self._build_dataset(json_files)

        if not self.evaluating:
            self.sample_func = self._sample_w2v_batch
        else:
            self.sample_func = self._iterate

    def _encoding_func(self, x):
        return utils.encode_protein_as_one_hot_vector(x, self.max_sequence_length)

    def _sample_w2v_batch(self, idx):

        target_sequence = self.sequences[int(np.random.rand()*len(self.sequences))]
        # grab a random sequence
        x = self.sequences_and_labels[target_sequence] #... and all of the
        # labels that come along with it (pfam ids)

        target_family = x[int(np.random.rand()*len(x))]
        # choose targets with probability proportional
        # to the number of sequences in that family

        y = self.labels_and_sequences[target_family]
        context_sequence = y[int(np.random.rand()*len(y))]

        # k, now sample self.n_negative_samples
        negative_examples = []
        i = 0
        while len(negative_examples) < self.n_negative_samples:
            negative_idx = np.array([int(np.random.rand()*len(self.pfam_names)) for _ in
                   range(self.n_negative_samples)])

            for idx in negative_idx:
                x = self.pfam_names[idx]
                if len(x) > 1:
                    negative_family = x[int(np.random.rand()*len(x))]
                else:
                    negative_family = x[0]

                if target_family != negative_family:
                    negative_examples.append(negative_family)

        if len(negative_examples) > self.n_negative_samples:
            negative_examples = negative_examples[:self.n_negative_samples]

        negatives = []
        for negative in negative_examples:
            x = self.labels_and_sequences[negative][int(np.random.rand()*len(self.labels_and_sequences[negative]))]
            negatives.append(x)

        target = torch.tensor(self._encoding_func(target_sequence))
        context = torch.tensor(self._encoding_func(context_sequence))
        negatives = [torch.tensor(self._encoding_func(x)) for x in negatives]
        labels = [1]
        labels.extend([0]*self.n_negative_samples)
        labels = torch.tensor([labels])
        return (target.float(), context.float(), negatives,
                        labels.float())

    def _iterate(self, idx):
        seq, labels = self.sequences[idx], self.pfam_names[idx]
        return torch.tensor(self._encoding_func(seq)), labels

    def _build_dataset(self, json_file):

        with open(json_file, 'r') as src:
            self.sequences_and_labels = json.load(src)

        self.labels_and_sequences = defaultdict(list)
        for prot_seq, accession_ids in self.sequences_and_labels.items():
            for i in accession_ids:
                self.labels_and_sequences[i].append(prot_seq)

        self.sequences = list(self.sequences_and_labels.keys())
        self.pfam_names = list(self.sequences_and_labels.values())

    def __len__(self):
        return len(self.sequences_and_labels)

    def __getitem__(self, idx):
        x = self.sample_func(idx)
        return x


class ProteinSequenceDataset(torch.utils.data.Dataset):

    def __init__(self,
            json_files,
            max_sequence_length,
            encode_as_image,
            multilabel,
            name_to_label_mapping,
            n_classes=None
            ):

        self.max_sequence_length = max_sequence_length
        self.multilabel = multilabel
        self.encode_as_image = encode_as_image
        self.name_to_label_mapping = name_to_label_mapping

        if n_classes is None:
            self.n_classes = utils.get_n_classes(self.name_to_label_mapping)
        else:
            self.n_classes = n_classes

        self._build_dataset(json_files)

    def _encoding_func(self, x):
        # TODO: implement more logic here to use variable encodings.

        labels, seq = x

        oh = encode_protein_as_one_hot_vector(seq, self.max_sequence_length)
        if not self.encode_as_image:
            oh = np.argmax(oh, axis=-1)

        if self.multilabel:
            label = np.zeros((self.n_classes, ))
            label[np.asarray(labels)] = 1
            labels = label

        return [oh, label]


    def _build_dataset(self, json_files):
        if not isinstance(json_files, list):
            json_files = [json_files]

        if len(json_files) == 1:
            self.sequences_and_labels = read_sequences_from_json(json_files[0], self.name_to_label_mapping)
        else:
            self.sequences_and_labels = []
            for j in json_files:
                self.sequences_and_labels.extend(read_sequences_from_json(j,
                    self.name_to_label_mapping))

        shuffle(self.sequences_and_labels)


    def __len__(self):
        return len(self.sequences_and_labels)

    def __getitem__(self, idx):

        x, y = self._encoding_func(self.sequences_and_labels[idx])
        return torch.tensor(x.squeeze()).transpose(-1, -2).float(), torch.tensor(y) 




if __name__ == '__main__':

    import matplotlib.pyplot as plt

    dirs = ['profmark0.2','profmark0.3','profmark0.4','profmark0.5','profmark0.6','profmark0.7',
          'profmark0.8','profmark0.9']

    root ='../../data/pmark-outputs/profmark0.6/json/train-sequences-and-labels.json' 
    dset = Word2VecStyleDataset(root, None,
            '../../data/pmark-outputs/profmark0.6/json/name-to-label.json')

    dset = torch.utils.data.DataLoader(dset, batch_size=1024,
            collate_fn=utils.pad_sequences_to_max_length_in_batch)
    i = 0
    s = time.time()
    for x in dset:
        print(x[0].shape)
    print(time.time() - s)
