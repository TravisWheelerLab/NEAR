import pdb
from collections import defaultdict

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
dataset = utils.RealisticAliPairGenerator()
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

with torch.no_grad():
    for seed_num in range(num_families):
        # generate a new sequence
        seq = utils.generate_sequences(
            1, len_generated_seqs, utils.amino_distribution
        ).squeeze()
        # add the raw form to the seed sequence (mutated later for evaluation)
        seed_sequences.append(seq)
        # transform into an encoding recognizable to the model
        seq = "".join([utils.char_to_index[c.item()] for c in seq])
        embedding = model(
            torch.as_tensor(utils.encode_protein_as_one_hot_vector(seq))
            .unsqueeze(0)
            .float()
            .to(dev)
        )
        # add our embedding to a database (a list in this case)
        embeddings.append(embedding.squeeze())
        # add len(embeddings) of the same labels to our label database, since each consensus sequence has
        # n embeddings that represent it
        labels.extend([seed_num] * embedding.shape[-1])

# concatenate embeddings and transpose contiguously
# so faiss can accept them
embeddings = torch.cat(embeddings, dim=-1).T.contiguous()
# create an index for easy searching
index = utils.create_logo_index(embeddings, embed_dim, dev)
# make labels into an array for easy np ops
labels = np.asarray(labels)

# set up data structures to record data
correct = defaultdict(int)
total_unique_labels = defaultdict(int)
subsitution_dists = utils.generate_sub_distributions()

n_seq = 0

with torch.no_grad():
    for i in range(num_families):
        # iterate over families
        seq = seed_sequences[i]
        # label is the index of the family
        label = i
        for n in range(n_mutations_per_sequence):
            n_seq += 1
            # mutate the sequence
            mutated_seq = utils.mutate_sequence(
                seq,
                int(sub_rate * len_generated_seqs),
                int(indel_rate * len_generated_seqs),
                sub_distributions=subsitution_dists,
                aa_dist=utils.amino_distribution,
            )

            mutated_seq = "".join([utils.char_to_index[c.item()] for c in mutated_seq])
            # encode it and feed it into the model
            mutated_seq = utils.encode_protein_as_one_hot_vector(mutated_seq)
            mutated_seq = torch.as_tensor(mutated_seq).to(dev).unsqueeze(0).float()

            #  comes out as 1xembed_dimxlen_embedding, so transpose it
            embedding = model(mutated_seq).squeeze().T.contiguous()
            # search it against the index, get distances between amino acids
            # and the indices of the nearest neighbors for each query embedding
            D, topn = index.search(embedding, k=20)
            # D is then len_embeddingxk
            # remove extra dims and ravel
            D = D.squeeze().ravel()
            topn = topn.squeeze().ravel()
            # sort the indices of top matches in the index by (descending) distance
            topn = topn[torch.argsort(D, descending=True)].cpu().numpy()
            # get the predicted classes by matching them to the vector of labels
            predicted_classes = [labels[j] for j in topn]
            for n in [1, 10, 20, 100, 200, 250]:
                # now, for topN in the list above, get whether or not the correct label
                # is in the predicted classes at that threshold
                correct[n] += label in predicted_classes[:n]
                # also get the number of unique classes predicted at that threshold
                # (so we can see our false-positive rate)
                x = len(set(predicted_classes[:n]))
                total_unique_labels[n] += x

print(f"{sub_rate}, {indel_rate}")
print(correct)
print("Percent correct @ different thresholds:")
print(",".join([str(s) for s in np.asarray(list(correct.values())) / n_seq]))
print("Average number of unique matches at threshold")
zz = np.asarray(list(total_unique_labels.values())) / n_seq
print(zz)
print("Average % filtration @ thresholds")
zz = (np.asarray(list(total_unique_labels.values())) / n_seq) / num_families
print(zz)
print("=========")
