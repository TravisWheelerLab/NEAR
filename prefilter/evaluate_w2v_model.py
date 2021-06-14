import os
import time
import json
import pdb
import numpy as np
import torch
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from collections import defaultdict

import utils.utils as u
from utils.datasets import Word2VecStyleDataset
import models as m
import losses as l

def collate_fn(batch):
    features = torch.stack([b[0] for b in batch])
    labels = []
    for b in batch:
        labels.append(b[1])
    return features, labels

def cosine_sim(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_embeddings_per_family(dataset, model, device='cuda'):

    name_to_embed = defaultdict(list)
    for batch in dataset:
        features, mask, labels = batch
        features = features.to(device)
        mask = mask.to(device)
        embedding = model(features, mask).to(device)
        for i, label_set in enumerate(labels):
            for family in label_set:
                name_to_embed[family].append(embedding[i])

    return name_to_embed

def get_closest(target, others):
    pass

# i = 0
# s = time.time()

if __name__ == '__main__':

    root = '../data/subset-for-overfitting/json/'
    name_to_label_mapping = root + 'name-to-label.json'
    test = root + 'test-subset.json'
    train = root + 'train-subset.json'

    test_dset = Word2VecStyleDataset(test, None, name_to_label_mapping,
            n_negative_samples=5, evaluating=True)

    train_dset = Word2VecStyleDataset(train, None, name_to_label_mapping,
            n_negative_samples=5, evaluating=True)

    test = torch.utils.data.DataLoader(test_dset, batch_size=32,
            collate_fn=u.pad_batch)
    train = torch.utils.data.DataLoader(train_dset, batch_size=32,
            collate_fn=u.pad_batch)

    arg_dict = {}
    model = m.Prot2Vec(m.PROT2VEC_CONFIG,
           arg_dict,
           evaluating=True).to('cuda')

    model.load_state_dict(torch.load('overfit-on-subset.pt'))

    test_to_embed = get_embeddings_per_family(test, model)
    train_to_embed = get_embeddings_per_family(train, model)
    test_labels = list(test_to_embed.keys())
    train_labels = list(train_to_embed.keys())

    intersection = set(train_labels).intersection(set(test_labels))


    tot = 0
    no = 0
    for protein_family in intersection:
        test_embeddings = test_to_embed[protein_family]
        for test_embedding in test_embeddings:
            max_sim = 0
            for train_family, train_embed in train_to_embed.items():
                if train_family not in intersection:
                    continue
                for train_embedding in train_embed:
                    c = cosine_sim(test_embedding, train_embedding) 
                    if c > max_sim:
                        max_sim = c
                        closest_match = train_family

            tot += closest_match == protein_family
            no += closest_match != protein_family
    print(tot)















