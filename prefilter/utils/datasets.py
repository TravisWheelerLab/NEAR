import os
import json
import torch
import numpy as np

import pdb
import time

from glob import glob
from random import shuffle
from collections import defaultdict

import utils as utils

__all__ = ['Word2VecStyleDataset',
           'ProteinSequenceDataset',
           ]


class Word2VecStyleDataset(torch.utils.data.Dataset):
    
    def __init__(self,
            json_files,
            max_sequence_length,
            n_negative_samples,
            evaluating=False
            ):

        self.max_sequence_length = max_sequence_length
        self.n_negative_samples = n_negative_samples
        self.evaluating = evaluating
        self._build_dataset(json_files)

        if self.evaluating:
            self.sample_func = self._iterate
        else:
            self.sample_func = self._sample_w2v_batch

    def _encoding_func(self, x):
        return utils.encode_protein_as_one_hot_vector(x, self.max_sequence_length)

    def _sample_w2v_batch(self, idx):

        set_of_positive_labels = []

        while len(set_of_positive_labels) == 0:

            target_sequence = self.sequences[int(np.random.rand()*len(self.sequences))]
            # grab a random sequence
            set_of_positive_labels = self.sequences_and_labels[target_sequence] #... and all of the
        # i need to figure out why some sequences have 0 labels associated with
        # them...

        # labels that come along with it (pfam ids)

        target_family = set_of_positive_labels[int(np.random.rand()*len(set_of_positive_labels))]
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
                elif len(x) == 1:
                    negative_family = x[0]
                else:
                    # this shouldn't happen. But it only happens a few
                    # times.
                    continue

                if negative_family not in set_of_positive_labels:
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

    def _build_dataset(self, json_files):

        if not isinstance(json_files, list):
            json_files = [json_files]
        self.sequences_and_labels = {}
        for f in json_files:
            with open(f, 'r') as src:
                dct = json.load(src)
                self.sequences_and_labels.update(dct)

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

    root = '/home/tom/pfam-carbs/small-dataset/'
    root = glob(os.path.join(root, "*train.json"))
    dset = Word2VecStyleDataset(root, None, 5)

    dset = torch.utils.data.DataLoader(dset, batch_size=32,
            collate_fn=utils.pad_word2vec_batch)
    i = 0

    s = time.time()
    cnt = 0
    for x in dset:
        cnt += 1
        print(x[0].shape)
        pass
        
    print(time.time() - s, (time.time()-s)/cnt)
