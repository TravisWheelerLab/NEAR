import torch
from sys import stdout
import numpy as np
import os
from glob import glob
from sklearn.neighbors import BallTree
import yaml

from prefilter.models import ResNet2d, ResNet1d, SupConLoss
from prefilter.utils import (
    PROT_ALPHABET,
    fasta_from_file,
    msa_from_file,
    encode_protein_as_one_hot_vector,
    encode_msa,
    logo_from_file,
)

# dataset of N pairs:
# anchor + positive, anchor + N negatives
# start w/ BCE

logo_path = "/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/"
fasta_path = "/home/tc229954/data/prefilter/pfam/seed/model_comparison/training_data_no_evalue_threshold/200_file_subset"

fasta_files = glob(os.path.join(fasta_path, "*-valid.fa"))
fasta_set = set()
name_to_sequences = {}
for fasta in fasta_files:
    bs = os.path.basename(fasta)
    bs = bs.split(".0.5")[0]
    fasta_set.add(bs)
    labels, sequences = fasta_from_file(fasta)
    name_to_sequences[bs] = [encode_protein_as_one_hot_vector(s) for s in sequences]

logo_files = glob(os.path.join(logo_path, "*.logo"))
# only going to eval at the end
logo_train = []
i = 0
names = []
name_to_logo = {}

for logo in logo_files:
    bs = os.path.basename(logo)
    bs = bs.split(".0.5")[0]
    if bs in fasta_set:
        logo_train.append(logo)
        names.append(bs)
        name_to_logo[bs] = logo_from_file(logo)


def sample_pos():
    name_idx = int(np.random.rand() * len(names))
    name = names[name_idx]
    pos_seqs, pos_logo = name_to_sequences[name], name_to_logo[name]
    seq_idx = int(np.random.rand() * len(pos_seqs))
    pos_seq = pos_seqs[seq_idx]
    return torch.as_tensor(pos_seq).unsqueeze(0), torch.as_tensor(pos_logo).unsqueeze(0)


def sample_neg():
    neg_idx_seq = int(np.random.rand() * len(names))
    neg_idx_logo = int(np.random.rand() * len(names))
    while neg_idx_logo == neg_idx_seq:
        neg_idx_logo = int(np.random.rand() * len(names))
    neg_seqs, neg_logo = (
        name_to_sequences[names[neg_idx_seq]],
        name_to_logo[names[neg_idx_logo]],
    )
    neg_seq_idx = int(np.random.rand() * len(neg_seqs))
    neg_seq = neg_seqs[neg_seq_idx]
    return torch.as_tensor(neg_seq).unsqueeze(0), torch.as_tensor(neg_logo).unsqueeze(0)


device = "cuda:2"
encoder = ResNet1d().to(device)
optim = torch.optim.Adam(encoder.parameters(), lr=6e-5)
labels = torch.as_tensor([1, 0, 0, 0]).to(device).float()
i = 0
cos_sim = torch.nn.CosineSimilarity(dim=-1)

if not os.path.isfile("trained_encoder.pt"):

    for epoch in range(100):

        for batch_idx in range(200):

            optim.zero_grad()
            pos_seq, pos_logo = sample_pos()
            neg_seq, neg_logo = sample_neg()

            # anchor
            pos_seq_embed = encoder(pos_seq.to(device).float()).squeeze()
            # positive
            pos_logo_embed = encoder(pos_logo.to(device).float()).squeeze()
            # negative
            neg_logo_embed = encoder(neg_logo.to(device).float()).squeeze()

            # a should be small
            a = 1 - cos_sim(pos_seq_embed, pos_logo_embed).float().to(device)
            # b should be large
            b = 1 - cos_sim(pos_seq_embed, neg_logo_embed).float().to(device)

            z = a - b + 1
            loss = torch.max(torch.tensor(0).to(device), z)

            # print(a.item(), b.item(), c.item(), d.item())
            if i % 20 == 0:
                print(f"{loss.item():.3f}, {a.item():.3f}, {b.item():.3f}\r")
            i += 1
            loss.backward()
            optim.step()

    torch.save(encoder, "trained_encoder.pt")
else:
    encoder = torch.load("trained_encoder.pt", map_location=torch.device(device))

# eval. all logos
logos = [name_to_logo[n] for n in names]
logo_embeddings = np.zeros((len(logos), 256))
with torch.no_grad():
    for i, logo in enumerate(logos):
        embedding = encoder(torch.as_tensor(logo).to(device).float().unsqueeze(0))
        logo_embeddings[i] = embedding.squeeze().detach().cpu().numpy()

# ball tree doesn't natively support cosine similarity as a metric,
# so just do the matmul for now.
logo_embeddings = torch.as_tensor(logo_embeddings).to(device).float()
logo_embeddings = torch.nn.functional.normalize(logo_embeddings, dim=-1)

total = 0
correct = 0
with torch.no_grad():
    for i, name in enumerate(names):
        sequences = name_to_sequences[name]
        total += len(sequences)
        for sequence in sequences:
            predicted_embedding = encoder(
                torch.as_tensor(sequence).to(device).float().unsqueeze(0)
            )
            predicted_embedding = predicted_embedding / torch.norm(
                predicted_embedding, dim=-1
            )
            nearest_neighbors = torch.matmul(
                logo_embeddings, predicted_embedding.T
            ).squeeze()
            nearest_neighbors_idx = torch.argsort(nearest_neighbors).cpu().numpy()
            if i in set(nearest_neighbors_idx[-5:]):
                correct += 1

print(correct, total, correct / total)
