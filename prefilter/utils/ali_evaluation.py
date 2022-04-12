import pdb

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import yaml

from prefilter.utils import (
    load_model,
    create_logo_index,
    create_parser,
    AliPairGenerator,
    NonDiagonalAliPairGenerator,
)
import prefilter.utils as utils


def mutate_seq(sequence, n_inserts=10, n_deletions=10):
    string_aa = [utils.INVERSE_PROT_MAPPING[i] for i in np.argmax(sequence, axis=0)]
    for _ in range(n_inserts):
        # insertion step
        pos = np.random.randint(0, len(string_aa))
        random_aa = alphabet[np.random.randint(0, len(alphabet))]
        string_aa.insert(pos, random_aa)
    for _ in range(n_deletions):
        # del step
        pos = np.random.randint(0, len(string_aa))
        string_aa.pop(pos)
    seq = utils.encode_protein_as_one_hot_vector("".join(string_aa))
    return seq


parser = create_parser()
parser.add_argument("--mutate", action="store_true")
parser.add_argument("--diag", action="store_true")

args = parser.parse_args()

logo_path = args.logo_path
hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
figure_path = args.figure_path
embed_dim = args.embed_dim
batch_size = args.batch_size
n_top = args.n_top
add_all_logos = args.add_all_families
benchmarking = args.benchmark
mutate = args.mutate

dev = "cuda" if torch.cuda.is_available() else "cpu"

with open(hparams_path, "r") as src:
    hparams = yaml.safe_load(src)

model = load_model(model_path, hparams, dev)
if args.diag:
    dataset = AliPairGenerator()
else:
    dataset = NonDiagonalAliPairGenerator()

embeddings = []
labels = []

for seed_num in range(dataset.num_seeds):
    seq = dataset[seed_num]
    seq = seq[0]
    seq = torch.as_tensor(seq).unsqueeze(0).to(dev).float()
    embedding = model(seq)
    embeddings.append(embedding.squeeze())
    labels.extend([seed_num] * embedding.shape[-1])

embeddings = torch.cat(embeddings, dim=-1).T.contiguous()
index = create_logo_index(embeddings, embed_dim, dev)
labels = np.asarray(labels)
alphabet = list(utils.PROT_ALPHABET.keys())


n_seq = 10000
from collections import defaultdict

correct = defaultdict(int)
total_unique_labels = defaultdict(int)

for i in range(n_seq):
    seq = dataset[i]
    label = seq[-1]
    seq = seq[0]
    correct_idx = set(np.where(labels == label)[0])
    if mutate:
        seq = mutate_seq(sequence=seq)
    seq = torch.as_tensor(seq).to(dev).unsqueeze(0).float()
    embedding = model(seq).squeeze().T.contiguous()
    D, topn = index.search(embedding, k=20)
    D = D.squeeze().ravel()
    topn = topn.squeeze().ravel()
    topn = topn[torch.argsort(D, descending=True)].cpu().numpy()
    predicted_classes = [labels[j] for j in topn]
    for n in [1, 10, 20, 100, 200, 250]:
        correct[n] += label in predicted_classes[:n]
        x = len(set(predicted_classes[:n]))
        total_unique_labels[n] += x

print(correct)
print(np.asarray(list(correct.values())) / n_seq)
zz = np.asarray(list(total_unique_labels.values())) / n_seq
# avg unique labels
print(zz)
