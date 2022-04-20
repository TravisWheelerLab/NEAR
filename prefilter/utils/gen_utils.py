import torch
import numpy as np

__all__ = [
    "generate_sequences",
    "mutate_sequence",
    "mutate_sequences",
    "amino_distribution",
    "char_to_index",
    "amino_alphabet",
    "generate_sub_distributions",
]

blsm_str = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""

amino_alphabet = [c for c in "ARNDCQEGHILKMFPSTWYV"]
char_to_index = {c: i for i, c in enumerate(amino_alphabet)}

amino_frequencies = torch.tensor(
    [
        0.074,
        0.042,
        0.044,
        0.059,
        0.033,
        0.058,
        0.037,
        0.074,
        0.029,
        0.038,
        0.076,
        0.072,
        0.018,
        0.040,
        0.050,
        0.081,
        0.062,
        0.013,
        0.033,
        0.068,
    ]
)

amino_distribution = torch.distributions.categorical.Categorical(amino_frequencies)
assert len(amino_alphabet) == len(amino_frequencies)


def blossum_to_probabilities(blossum_string):
    blossum_mat = torch.zeros(20, 20)
    sub_mat = torch.zeros(20, 20)
    str_mat = blossum_string.split("\n")

    for i in range(20):
        str_row = str_mat[i].split(" ")
        for j in range(20):
            val = float(str_row[j])
            blossum_mat[i, j] = val
            sub_mat[i, j] = val

        sub_mat[i, i] = -10000.0

    return torch.softmax(blossum_mat, dim=-1), torch.softmax(sub_mat, dim=-1)


def blossum_mat():
    return blossum_to_probabilities(blsm_str)


def generate_sub_distributions():
    blsm_mat, sub_mat = blossum_mat()

    sub_distributions = []
    for i in range(20):
        sub_distributions.append(
            torch.distributions.categorical.Categorical(sub_mat[i])
        )

    return sub_distributions


def generate_sequences(num_sequences, length, aa_dist):
    return aa_dist.sample((num_sequences, length))


def mutate_sequence(
    sequence, labelvec, substitutions, indels, sub_distributions, aa_dist
):
    # we can probably go into fourier space for computing indels efficiently
    # but that sounds like a lot of work
    # and it might not even work

    # alternatively you can do the index tricks for massive shifting
    # which is probably still faster
    # and works for batches

    # but instead we slow it down here

    seq = sequence.clone()
    sub_indices = torch.randperm(len(seq))[:substitutions]

    for i in range(len(sub_indices)):
        seq[sub_indices[i]] = sub_distributions[seq[sub_indices[i]]].sample()

    seq = seq.tolist()

    deletion_indices = torch.randperm(len(seq))[:indels]
    insertion_indices = torch.randperm(len(seq))[:indels]
    insertion_aminos = generate_sequences(1, indels, aa_dist=aa_dist)[0]

    for i in range(indels):
        seq.pop(deletion_indices[i])
        labelvec.pop(deletion_indices[i])
        seq.insert(insertion_indices[i], insertion_aminos[i])
        labelvec.insert(insertion_indices[i], np.max(labelvec) + 1)

    seq = torch.tensor(seq)

    return seq, labelvec


def mutate_sequences(sequences, substitutions, indels):
    mutated_sequences = []
    for i in range(sequences.shape[0]):
        mutated_sequences.append(mutate_sequence(sequences[i], substitutions, indels))
    return torch.stack(mutated_sequences, dim=0)


if __name__ == "__main__":
    x = generate_sequences(10, 10, amino_distribution)
    print(x.shape)
