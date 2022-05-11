import numpy as np
import os
from glob import glob
import prefilter.utils as utils

train_files = glob("/home/tc229954/data/prefilter/pfam/seed/clustered/0.8/*-train.fa")

print(len(train_files))
train_seqs = []
min_seq_len = 256
cluster_reps = 1
seed_sequences = []

for file in train_files:
    valid_file = file.replace("-train.fa", "-valid.fa")
    if os.path.isfile(valid_file):
        headers, seqs = utils.fasta_from_file(file)
        seqs = [
            s.replace(".", "")[:min_seq_len]
            for s in seqs
            if len(s.replace(".", "")) >= min_seq_len
        ]
        if len(seqs):
            seed_sequences.extend(seqs[:cluster_reps])

print(len(seed_sequences))
print(sum(list(map(len, seed_sequences))))
