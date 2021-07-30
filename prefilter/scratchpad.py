import os
import torch
from glob import glob
from argparse import ArgumentParser

import models as m
import utils.utils as utils
from evaluate_w2v_model import SimpleSequenceIterator, get_embeddings_per_sequence, get_mean_embedding_per_family

if __name__ == '__main__':

    device = 'cuda'

    consensus_sequences = glob("../data/consensus_sequences/*fa")
    train_sequences = glob("../data/small-dataset/*train*")
    
    model_path = './with-normalization-small-dataset-2files-500ep.pt'
    # model_path = './with-normalization-small-dataset-50.pt'


    conf = m.PROT2VEC_CONFIG
    conf['normalize'] = True

    model = m.Prot2Vec(conf, evaluating=True)
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)
    model.eval()

    consensus_sequences = SimpleSequenceIterator(fasta_files=consensus_sequences)
    train_sequences = SimpleSequenceIterator(json_files=train_sequences)

    train_sequences = torch.utils.data.DataLoader(train_sequences,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  collate_fn=utils.pad_batch)

    consensus_sequences = torch.utils.data.DataLoader(consensus_sequences,
                                                      batch_size=32,
                                                      shuffle=False,
                                                      collate_fn=utils.pad_batch)
    
    consensus_families, consensus_embeddings = get_embeddings_per_sequence(consensus_sequences, 
            model,
            device=device)

    train_families, train_mean_embeddings = get_mean_embedding_per_family(train_sequences,
            model,
            device=device)

    single_sequence_train, single_sequence_embeddings = get_embeddings_per_sequence(train_sequences, 
            model, device=device)

    import pdb
    pdb.set_trace()


