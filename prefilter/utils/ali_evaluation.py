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
)
import prefilter.utils as utils

args = create_parser().parse_args()

logo_path = args.logo_path
hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
figure_path = args.figure_path
embed_dim = args.embed_dim
batch_size = args.batch_size
n_top = args.n_top
add_all_logos = args.add_all_families
benchmarking = args.benchmark

dev = "cuda" if torch.cuda.is_available() else "cpu"

with open(hparams_path, "r") as src:
    hparams = yaml.safe_load(src)

model = load_model(model_path, hparams, dev)
dataset = AliPairGenerator()
embeddings = []
labels = []

for seed_num in range(dataset.num_seeds):
    seq, _, _ = dataset[seed_num]
    seq = torch.as_tensor(seq).unsqueeze(0).to(dev).float()
    embedding = model(seq)
    embeddings.append(embedding.squeeze())
    labels.extend([seed_num] * embedding.shape[-1])

embeddings = torch.cat(embeddings, dim=-1).T.contiguous()
index = create_logo_index(embeddings, embed_dim, dev)
labels = np.asarray(labels)
alphabet = list(utils.PROT_ALPHABET.keys())

for i in range(10):
    # x = []
    # for _ in range(2):
    seq, _, label = dataset[i]
    correct_idx = set(np.where(labels == label)[0])
    #     string_aa = [utils.inverse[i] for i in np.argmax(seq, axis=0)]
    #     original = string_aa.copy()
    #     for _ in range(10):
    #         pos = np.random.randint(0, len(string_aa))
    #         random_aa = alphabet[np.random.randint(0, len(alphabet))]
    #         string_aa.insert(pos, random_aa)
    #         # insertion step
    #     for _ in range(10):
    #         pos = np.random.randint(0, len(string_aa))
    #         string_aa.pop(pos)
    #         # del step
    # seq = torch.as_tensor(utils.encode_protein_as_one_hot_vector("".join(string_aa))).unsqueeze(0).to(dev).float()
    seq = torch.as_tensor(torch.as_tensor(seq).to(dev).unsqueeze(0).float())
    embedding = model(seq).squeeze().T.contiguous()
    #     x.append(embedding.detach().cpu().numpy())
    # matmul = np.matmul(x[0].T, x[1])
    # plt.imshow(matmul)
    # plt.colorbar()
    # plt.savefig("indels.png", bbox_inches="tight")
    # plt.close()
    # print(matmul.shape)
    # exit()
    D, topn = index.search(embedding, k=1)
    set_of_labels_for_this_sequence = []
    misclassifications = []

    for AA in topn:
        s = set(AA.cpu().numpy()).intersection(correct_idx)
        miss = set(AA.cpu().numpy()).difference(correct_idx)
        set_of_labels_for_this_sequence.extend(list(s))
        misclassifications.extend(list(miss))

    print(len(set(set_of_labels_for_this_sequence)), len(set(misclassifications)))
