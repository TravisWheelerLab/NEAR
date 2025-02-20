# Standard library imports
import os
import random
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Tuple
from pathlib import Path
from Bio import SeqIO
import sys
import math
import time

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import yaml

# NEAR imports
from dataloader import AlignmentDataset
from models import NEARResNet, NEARUNet

def create_tensors_from_fasta(file_path, pad_power_two=False, min_seq_length = 32, alphabet=None):

    if alphabet is None:
        amino_acids = 'XARNDCEQGHILKMFPSTWYVBJZ0'
        alphabet = {aa: i for i, aa in enumerate(amino_acids)}
        alphabet['U'] = 0
        alphabet['B'] = 0
        alphabet['J'] = 0
        alphabet['0'] = 0

    sequence_tensors = []

    length_to_count = defaultdict(int)
    index_to_id = []
    index_to_length = []
    index_to_twoindex = []
    

    # First pass we just count sequence lengths
    for (i, record) in enumerate(SeqIO.parse(file_path, "fasta")):
        seq_length = len(record.seq)

        # Check sequence length validity
        if seq_length < min_seq_length:
            print(str(i) + ": Sequence is too small, ID=" + record.id)
            print("Terminating.")
            exit(-1)
            continue
        # Pad sequence length
        if pad_power_two:
            # If this condition is true, then seq_length is a power of 2
            if (seq_length & (seq_length - 1)) != 0:
                seq_length = 2**math.ceil(math.log2(seq_length))
        
        length_to_count[seq_length] += 1
        index_to_id.append(record.id)
        index_to_twoindex.append((seq_length, length_to_count[seq_length] - 1))

    # Create tensors
    sequence_tensors = dict()
    for length in length_to_count:
        count = length_to_count[length]
        tensor = torch.zeros(count, length, dtype=int)
        sequence_tensors[length] = tensor

    # Read sequences into tensors
    for (i, record) in enumerate(SeqIO.parse(file_path, "fasta")):
        seq_length = len(record.seq)
        index_to_length.append(seq_length)
        if pad_power_two:
            if (seq_length & (seq_length - 1)) != 0:
                seq_length = 2**math.ceil(math.log2(seq_length))
        
        tensor = sequence_tensors[seq_length]
        _, idx = index_to_twoindex[i]

        seq = str(record.seq).upper()
        tensor[idx] = torch.tensor(list(seq.encode('ascii')), dtype=int)

    for key in sequence_tensors.keys():
        for c in alphabet.keys():
            sequence_tensors[key][sequence_tensors[key] == ord(c)] = alphabet[c]

    return sequence_tensors, index_to_twoindex, index_to_id, index_to_length

def embed_sequence_tensors(model, sequence_tensors, device, batch_size=512):
    embeddings = dict()
    for key in sequence_tensors:
        with torch.no_grad():
            #print(sequence_tensors[key].shape)
            if len(sequence_tensors[key]) <= batch_size:
                embeddings[key] = model(sequence_tensors[key].to(device))
            else:
                tensor = sequence_tensors[key]
                num_sequences = tensor.size(0)
                num_batches = (num_sequences + batch_size - 1) // batch_size
                embeddings[key] = []
                
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_sequences)
                    batch = tensor[start_idx:end_idx].to(device)
                    batch_embedding = model(batch)
                    batch_embedding = batch_embedding.cpu()
                    embeddings[key].append(batch_embedding)
                
                embeddings[key] = torch.cat(embeddings[key], dim=0)
            embeddings[key] = torch.transpose(embeddings[key], -1, -2).cpu().numpy()


    return embeddings

def save_embeddings(embeddings, index_to_twoindex, index_to_id, index_to_length, output_path):
    embedding_list = []
    for i, (x,y) in enumerate(index_to_twoindex):
        embedding_list.append(embeddings[x][y,:index_to_length[i]])
    
    np.savez(output_path, **{str(i): array for i, array in enumerate(embedding_list)})
    


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


global_start_time = time.time()

if len(sys.argv) != 5:
    print("Usage: python near_embed.py <config_path> <parameter path> <fasta_path> <output_path>")
    sys.exit(1)

config_path     = sys.argv[1]
parameter_path  = sys.argv[2]
fasta_path      = sys.argv[3]
output_path     = sys.argv[4]


models = {'UNet': NEARUNet, 'ResNet': NEARResNet}

print("Reading config file: " + config_path)
config = load_config(config_path)

print("Creating model from config " + config_path)
model = models[config['model']](**config['model_args'])


print("Loading model parameters from " + parameter_path)

model_state_dict = torch.load(parameter_path, map_location='cpu', weights_only=True)['model_state_dict']
model.load_state_dict(model_state_dict)

device = 'cuda'
if 'device' in config:
    device = torch.device(config['device'])

print("Pushing model to " + str(device))
model.to(device)
model.eval()

print("Creating sequence tensors from FASTA " + fasta_path)
sequence_tensors, index_to_twoindex, index_to_id, index_to_length = create_tensors_from_fasta(fasta_path, 
                          pad_power_two=config['model'] == 'UNet')

print("Embedding tensors... ")

start_time = time.time()
embeddings = embed_sequence_tensors(model, sequence_tensors, device)
end_time = time.time()

print("Done. Embedding time (S) = " + str(end_time - start_time))
print("Saving embeddings")

save_embeddings(embeddings, index_to_twoindex, index_to_id, index_to_length, output_path)


global_end_time = time.time()
print("Done! Total time: " + str(global_end_time - global_start_time))

