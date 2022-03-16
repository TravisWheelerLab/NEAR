import torch
import numpy as np
import os
from glob import glob
import yaml

from prefilter.models import ResNet2d, ResNet1d, SupConLoss
from prefilter.utils import (
    PROT_ALPHABET,
    fasta_from_file,
    msa_from_file,
    encode_protein_as_one_hot_vector,
    encode_msa,
)

# dataset of N pairs:
# anchor + positive, anchor + N negatives
# start w/ BCE

afa_path = "/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/"
fasta_path = "/home/tc229954/data/prefilter/pfam/seed/model_comparison/training_data_no_evalue_threshold/200_file_subset"

fasta_files = glob(os.path.join(fasta_path, "*-train.fa"))
fasta_set = set()
name_to_sequences = {}
for fasta in fasta_files:
    bs = os.path.basename(fasta)
    bs = bs.split(".0.5")[0]
    fasta_set.add(bs)
    labels, sequences = fasta_from_file(fasta)
    name_to_sequences[bs] = [encode_protein_as_one_hot_vector(s) for s in sequences]

afa_files = glob(os.path.join(afa_path, "*.afa"))
# only going to eval at the end
afa_train = []
i = 0
names = []
name_to_msa = {}

for afa in afa_files:
    bs = os.path.basename(afa)
    bs = bs.split(".0.5")[0]
    if bs in fasta_set:
        afa_train.append(afa)
        names.append(bs)
        labels, sequences = msa_from_file(afa)
        name_to_msa[bs] = encode_msa(sequences)


def sample_pos():
    name_idx = int(np.random.rand() * len(names))
    name = names[name_idx]
    pos_seqs, pos_msa = name_to_sequences[name], name_to_msa[name]
    seq_idx = int(np.random.rand() * len(pos_seqs))
    pos_seq = pos_seqs[seq_idx]
    return torch.as_tensor(pos_seq).unsqueeze(0), torch.as_tensor(pos_msa).unsqueeze(0)


def sample_neg():
    neg_idx_seq = int(np.random.rand() * len(names))
    neg_idx_msa = int(np.random.rand() * len(names))
    while neg_idx_msa == neg_idx_seq:
        neg_idx_msa = int(np.random.rand() * len(names))
    neg_seqs, neg_msa = (
        name_to_sequences[names[neg_idx_seq]],
        name_to_msa[names[neg_idx_msa]],
    )
    neg_seq_idx = int(np.random.rand() * len(neg_seqs))
    neg_seq = neg_seqs[neg_seq_idx]
    return torch.as_tensor(neg_seq).unsqueeze(0), torch.as_tensor(neg_msa).unsqueeze(0)


device = "cuda:3"
msa_encoder = ResNet2d().to(device)
sequence_encoder = ResNet1d().to(device)
optim = torch.optim.Adam(
    list(msa_encoder.parameters()) + list(sequence_encoder.parameters())
)
crit = torch.nn.BCEWithLogitsLoss()
labels = torch.as_tensor([1, 0, 0, 0]).to(device).float()

for epoch in range(1000):

    for batch_idx in range(200):

        optim.zero_grad()
        pos_seq, pos_msa = sample_pos()
        neg_seq, neg_msa = sample_neg()

        pos_seq_embed = torch.nn.functional.normalize(
            sequence_encoder(pos_seq.to(device).float()).squeeze(), dim=-1, p=2
        )
        pos_msa_embed = torch.nn.functional.normalize(
            msa_encoder(pos_msa.to(device).float()).squeeze(), dim=-1, p=2
        )
        neg_seq_embed = torch.nn.functional.normalize(
            sequence_encoder(neg_seq.to(device).float()).squeeze(), dim=-1, p=2
        )
        neg_msa_embed = torch.nn.functional.normalize(
            msa_encoder(neg_msa.to(device).float()).squeeze(), dim=-1, p=2
        )

        a = torch.dot(pos_seq_embed, pos_msa_embed).float().to(device)
        b = torch.dot(pos_seq_embed, neg_seq_embed).float().to(device)
        c = torch.dot(pos_seq_embed, neg_msa_embed).float().to(device)
        d = torch.dot(pos_msa_embed, neg_seq_embed).float().to(device)

        loss = crit(
            torch.tensor([a, b, c, d], requires_grad=True).to(device), labels.float()
        )
        print([a.item(), b.item()])
        loss.backward()
        print(loss.item())
        optim.step()
        del a
        del b
        del c
        del d
        del pos_seq_embed
        del pos_msa_embed
        del neg_msa_embed
        del neg_seq_embed
