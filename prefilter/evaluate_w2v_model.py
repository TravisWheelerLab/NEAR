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
                        descending=True).squeeze()

    if idx.ndim > 1:
        idx = idx[:, :n]
    else:
        idx = idx[:n]

    return target_names[idx.detach().cpu().numpy()]

def intersection_of_sets(matches, true_labels):

    x = len(set(matches).intersection(set(true_labels)))

    return x

if __name__ == '__main__':

    device = 'cuda'

    train_files = glob('/home/tom/pfam-carbs/small-dataset/*train.json')
    
    test_files = glob('../data/small-dataset/*test-split.json')

    train_files = ['../data/small-dataset/NlpE-train.json',
                   '../data/small-dataset/DUF627-0.5-train.json']
    test_files =  ['../data/small-dataset/NlpE-0.5-test-split.json',]
              #    '../data/small-dataset/DUF627-0.5-test-split.json']

    if len(test_files) == 0 or len(train_files) == 0:
        print('one of train or test had zero length. exiting')
        exit(0)

    model_path = './with-normalization-small-dataset-2files-500ep.pt'
    # model_path = './with-normalization-small-dataset-50.pt'

    conf = m.PROT2VEC_CONFIG
    conf['normalize'] = True

    model = m.Prot2Vec(conf, evaluating=True)
    model.load_state_dict(torch.load(model_path))

    model = model.cuda()
    model.eval()

    tot = 0
    for p in model.parameters():
        tot += torch.numel(p)

    test_dataset = SimpleSequenceIterator(test_files)
    train_dataset = SimpleSequenceIterator(train_files)

    test_data = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=1,
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
    intersection_bool_count = 0
    intersection_pct_count = 0
    top_n = 100

    with torch.no_grad():

        for features, mask, labels in test_data:


            query = model(features.cuda(), mask.cuda()).unsqueeze(0)

            match_names = get_closest_n_embeddings(query, 
                                     train_embeddings,
                                     train_families,
                                     top_n)

            query = query.detach().cpu().numpy()

            if match_names.ndim > 1:

                for true_labels, matches in zip(labels, match_names):
                    x = intersection_of_sets(matches, true_labels)
                    tot += 1
                    count += x / len(true_labels)

            else:

                labels = labels[0]
                x = intersection_of_sets(match_names, labels)
                tot += 1
                intersection_pct_count += x / len(labels)
                print('===')
                print(len(labels), set(match_names))
                print(labels)
                intersection_bool_count += 1 if x else 0


            # print(tot)
        s =  'sequences tested: {}, sequences in train: {}'
        s += '\nnumber of test sequences with their closest neighbor'
        s += '\nin train having at least one of the same labels'
        s += '\nin the top {} nearest neighbors: {}'
        s += '\nthe overlap b/t sets of labels between closest'
        s += '\ntrain and test embeddings is {} (best is {})'
        s = s.format(tot, train_embeddings.shape[0], top_n, 
                intersection_bool_count, intersection_pct_count,
                tot) 
        print(s)
