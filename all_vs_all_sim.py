import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

# take the query embeddings

with open("query_embeddings.txt", "rb") as src:
    embeddings = pickle.load(src)

# now do all-vs-all distance
diags = []
off_diags = []
for i in range(len(embeddings)):
    embed_i = embeddings[i]
    for j in range(i, len(embeddings)):
        # calculate distance
        distances = torch.cdist(embed_i, embeddings[j])
        if i == j:
            diags.append(distances.ravel())
        else:
            off_diags.append(distances.ravel())

# now, histogram the off-and-on diagonals
pdb.set_trace()
