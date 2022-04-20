import pdb
from glob import glob
from collections import defaultdict
from sys import stdout

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import yaml

import prefilter.utils as utils

parser = utils.create_parser()
parser.add_argument("--sub_rate", type=float)
parser.add_argument("--indel_rate", type=float)

args = parser.parse_args()

# get model files
hparams_path = os.path.join(args.model_root_dir, "hparams.yaml")
model_path = os.path.join(args.model_root_dir, "checkpoints", args.model_name)
embed_dim = args.embed_dim

dev = "cuda" if torch.cuda.is_available() else "cpu"

with open(hparams_path, "r") as src:
    hparams = yaml.safe_load(src)


# Set up model and dataset
model = utils.load_model(model_path, hparams, dev)
# RealisticPairGenerator is a wrapper around Daniel's code
# How many "consensus" sequences to generate?
num_families = 1000
# self-explanatory
len_generated_seqs = 256
# how many mutated sequence per consensus sequence to generate at evaluation time
n_mutations_per_sequence = 10
# substitution rate
sub_rate = args.sub_rate
# indel rate
indel_rate = args.indel_rate


embeddings = []
labels = []
seed_sequences = []

# afa_files = glob(
#     "/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/*-train.sto.afa"
# )
afa_files = glob(
    "/home/tc229954/data/prefilter/pfam/seed/training_data/1000_file_subset/*-train.fa"
)
# afa_files = glob("/home/tc229954/data/prefilter/pfam/seed/clustered/0.5/*-train.fa")

from random import shuffle, seed

seed(0)
shuffle(afa_files)

dataset = utils.AliEvaluator(afa_files, length_of_seq=None, pad=False)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, collate_fn=utils.pad_batch_with_labels
)

labels = []
num_unique = set()
with torch.no_grad():
    for label, ali in enumerate(dataset.alidb):
        target_seq = ali[0]
        num_unique.add(label)
        # generate a new sequence
        # add the raw form to the seed sequence (mutated later for evaluation)
        if len(target_seq) < 100:
            pad_len = 100 - len(target_seq)
            random_seq = utils.generate_sequences(1, pad_len, utils.amino_distribution)
            if pad_len != 1:
                random_seq = random_seq.squeeze()
            target_seq = target_seq + "".join(
                [utils.amino_alphabet[c.item()] for c in random_seq]
            )

        feat = (
            torch.tensor(utils.encode_protein_as_one_hot_vector("".join(target_seq)))
            .to(dev)
            .float()
            .unsqueeze(0)
        )
        embedding = model(feat)
        labels.extend([label] * embedding.shape[-1])
        # add len(embeddings) of the same labels to our label database, since each consensus sequence has
        # n embeddings that represent it
        embeddings.append(embedding.squeeze().T.contiguous())

# concatenate embeddings and transpose contiguously
# so faiss can accept them
num_families = len(num_unique)

embeddings = torch.cat(embeddings, dim=0).contiguous()
# create an index for easy searching
index = utils.create_logo_index(embeddings, embed_dim, dev)
# make labels into an array for easy np ops
labels = np.asarray(labels)
# eval.
dataloader.dataset.seed_seqs = False

# set up data structures to record data
correct = defaultdict(int)
total_unique_labels = defaultdict(int)
subsitution_dists = utils.generate_sub_distributions()

n_seq = 0

j = 0
with torch.no_grad():
    for features, features_mask, label in dataloader:
        stdout.write(f"{j/len(dataloader)*100:.3f}\r")
        j += 1
        embedding, mask = model(features.float().to(dev), features_mask.to(dev))
        embedding = ~mask.squeeze()[:, None, :] * embedding
        embedding = embedding.transpose(-1, -2)
        embedding = torch.cat(torch.unbind(embedding), dim=0)
        # D is then len_embeddingxk
        curr_char = 0
        n_chars = torch.sum(~mask.squeeze(), dim=1)
        D, topn = index.search(embedding, k=100)
        for i in range(len(label) - 1):
            # n_char is 0... means that there are no masked characters
            # in the sequence, so I just need to grab the
            n_char = n_chars[i]
            if n_char == 0:
                continue

            n_seq += 1
            sequence_embeddings = embedding[curr_char : curr_char + n_char]
            curr_char += n_char
            D_sub = D[curr_char : curr_char + n_char]
            topn_sub = topn[curr_char : curr_char + n_char]
            D_sub = D_sub.squeeze().ravel()
            topn_sub = topn_sub.squeeze().ravel()
            topn_sub = topn_sub[torch.argsort(D_sub, descending=True)].cpu().numpy()
            # get the predicted classes by matching them to the vector of labels
            predicted_classes = [labels[j] for j in topn_sub]
            lab = label[i]
            for n in [1, 10, 20, 100, 200, 250, 400]:
                # now, for topN in the list above, get whether or not the correct label
                # is in the predicted classes at that threshold
                correct[n] += lab in predicted_classes[:n]
                # also get the number of unique classes predicted at that threshold
                # (so we can see our false-positive rate)
                x = len(set(predicted_classes[:n]))
                total_unique_labels[n] += x

print()
print(f"Num correct @ thresholds: (total seq {n_seq}, total families: {num_families})")
print(",".join([f"{s}" for s in np.asarray(list(correct.values()))]))
print("Percent correct @ different thresholds:")
print(",".join([f"{s:.3f}" for s in np.asarray(list(correct.values())) / n_seq]))
print("Average number of unique matches at threshold")
zz = np.asarray(list(total_unique_labels.values())) / n_seq
print(",".join([f"{z:.3f}" for z in zz]))
print("Average % filtration @ thresholds")
zz = 1 - ((np.asarray(list(total_unique_labels.values())) / n_seq) / num_families)
print(",".join([f"{z:.3f}" for z in zz]))
