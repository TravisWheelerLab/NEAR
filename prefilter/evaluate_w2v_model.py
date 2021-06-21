import os
import time
import json
import pdb
import numpy as np
import torch

from collections import defaultdict
from glob import glob
from sys import stdout

import utils.utils as utils
import models as m
import losses as l

def cosine_sim(a, b):

    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

def _load_sequences_from_json(json_files):

    if not isinstance(json_files, list):
        json_files = [json_files]
    sequences_and_labels = {}
    for f in json_files:
        with open(f, 'r') as src:
            sequences_and_labels.update(json.load(src))

    return sequences_and_labels


class SimpleSequenceIterator(torch.utils.data.Dataset):

    def __init__(self, json_files):

        sequences_and_labels = _load_sequences_from_json(json_files)
        self.sequences = list(sequences_and_labels.keys())
        self.labels = list(sequences_and_labels.values())
        self.numel = len(sequences_and_labels)
        self.lengths = list(map(len, sequences_and_labels.keys()))
        self.idx = 0

        del sequences_and_labels

    def __len__(self):
        return self.numel

    def __getitem__(self, idx):
        features = utils.encode_protein_as_one_hot_vector(self.sequences[idx])
        return features, self.labels[idx]

def get_embeddings_per_family(dataloader, model, unwrap=False, device='cuda'):

    family_to_embedding = defaultdict(list)

    with torch.no_grad():

        for batch in dataloader:
            features, features_mask, labels = batch
            embeddings = model(features.to(device), features_mask.to(device))
            embeddings = embeddings.detach().cpu().numpy()
            for families, embedding in zip(labels, embeddings):
                for family in families:
                    family_to_embedding[family].append(embedding)

    families = []
    embeddings = []

    for family in family_to_embedding:
        embeddings.extend(np.stack(family_to_embedding[family]))
        families.extend([family]*len(family_to_embedding[family]))

    return np.array(families), np.array(embeddings)

def get_closest_n_embeddings(query, targets, target_names, n=100):
    '''
    query: vector of shape 1xembedding dim
    targets: matrix of shape n_targetsxembedding dim
    target_names: vector of strings of shape n_targetsx1
    '''

    normed_query = query / torch.linalg.norm(query, dim=1, keepdims=True)
    normed_targets = targets / torch.linalg.norm(targets, dim=1, keepdims=True)

    cosine_similarities = normed_query@normed_targets.T

    idx = torch.argsort(cosine_similarities, 
                        axis=1,
                        descending=True).squeeze()[:, :n]
    return target_names[idx.detach().cpu().numpy()]

def intersection_of_sets(matches, true_labels):
    x = len(set(matches).intersection(set(true_labels)))
    return x


if __name__ == '__main__':

    device = 'cuda'

    train_files = glob('/home/tom/pfam-carbs/1k/*train.json')
    test_files = glob('/home/tom/pfam-carbs/1k/*test-split.json')

    print(len(train_files), len(test_files))

    model_path = './overfit-for-testing-weekend.pt'

    model = m.Prot2Vec(m.PROT2VEC_CONFIG, True)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model = model.eval()

    test_dataset = SimpleSequenceIterator(test_files)
    train_dataset = SimpleSequenceIterator(train_files)

    test_data = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=32,
                                            shuffle=False,
                                            collate_fn=utils.pad_batch)

    train_data = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=32, shuffle=False,
                                            collate_fn=utils.pad_batch)


    train_families, train_embeddings = get_embeddings_per_family(train_data, 
            model)

    train_embeddings = torch.tensor(train_embeddings).to(device)

    tot = 0
    count = 0

    for features, mask, labels in test_data:

        with torch.no_grad():

            query = model(features.cuda(), mask.cuda()).unsqueeze(0)

        match_names = get_closest_n_embeddings(query, 
                                 train_embeddings,
                                 train_families,
                                 10000)

        query = query.detach().cpu().numpy()

        del query

        for true_labels, matches in zip(labels, match_names):

            x = intersection_of_sets(matches, true_labels)
            tot += 1
            count += x / len(true_labels)

        print(tot)
    print(tot, count, count / tot)
