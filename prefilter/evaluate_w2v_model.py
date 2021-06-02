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
import models as m
import losses as l

def collate_fn(batch):
    features = torch.stack([b[0] for b in batch])
    labels = []
    for b in batch:
        labels.append(b[1])
    return features, labels

if __name__ == '__main__':

    pmark = 0.7
    root = '../data/pmark-outputs/profmark{}/json/'.format(pmark)
    max_sequence_length = 256
    name_to_label_mapping = root + 'name-to-label.json'
    train_files = root + 'train-sequences-and-labels.json'
    test_files = root + 'test-sequences-and-labels.json'
    valid_files = root + 'val-sequences-and-labels.json'

    train = u.Word2VecStyleDataset(train_files,
                                   max_sequence_length,
                                   name_to_label_mapping,
                                   evaluating=True)
    test = u.Word2VecStyleDataset(test_files,
                                   max_sequence_length,
                                   name_to_label_mapping,
                                   evaluating=True)

    train = torch.utils.data.DataLoader(train, 
            batch_size=32, shuffle=True,
            collate_fn=collate_fn)
    test = torch.utils.data.DataLoader(test, 
            batch_size=32, shuffle=True,
            collate_fn=collate_fn)


    arg_dict = {}
    arg_dict['metrics'] = m.configure_metrics()
    arg_dict['test_files'] = test_files
    arg_dict['train_files'] = train_files
    arg_dict['valid_files'] = valid_files
    arg_dict['max_sequence_length'] = max_sequence_length
    arg_dict['name_to_label_mapping'] = name_to_label_mapping
    arg_dict['lr'] = 1e-3
    arg_dict['batch_size'] = 32
    arg_dict['num_workers'] = 1
    arg_dict['gamma'] = 1

    model = m.Prot2Vec(m.PROT2VEC_CONFIG, arg_dict)
    model.load_state_dict(torch.load('./first-pass-at-prot2vec.pt'))
    model.eval()
    model = model.cuda()

    pfam_id_to_mean_embedding = defaultdict(list)
    i = 0
    for x, y in train: 
        embeddings = model(x.cuda().float())
        for embedding, pfam_ids in zip(embeddings, y):
            for pfam_id in pfam_ids:
                pfam_id_to_mean_embedding[pfam_id].append(embedding.detach().cpu().numpy())
        i += 1
        if i > 0:
            break

    ls = list(map(len, list(pfam_id_to_mean_embedding.values())))
    pfam_id_to_mean_embedding_ = {}
    for k, v in pfam_id_to_mean_embedding.items():
        x = np.mean(np.stack(v), axis=0)
        pfam_id_to_mean_embedding_[k] = x

    mean_train = np.stack(list(pfam_id_to_mean_embedding_.values()))
    train_labels = list(pfam_id_to_mean_embedding_.keys())

    mean_train = mean_train / np.linalg.norm(mean_train, axis=0)
    i = 0
    num_correct = 0
    num_incorrect = 0
    for x, y in train:
        embeddings = model(x.cuda().float()).detach().cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=0)
        preds = np.matmul(embeddings, mean_train.T)
        preds = np.argmax(preds, axis=1)
        for pred_idx, yy  in zip(preds, y):
            num_correct += train_labels[pred_idx] in set(yy)
            num_incorrect += train_labels[pred_idx] not in set(yy)
            #if correct:
            #    #s = 'predicted to be (nn in cosine sim) {}, actually {}'
            #    #print(s.format(train_labels[pred_idx], yy))
            #    num_correct += 1

    print(num_correct, num_incorrect)
