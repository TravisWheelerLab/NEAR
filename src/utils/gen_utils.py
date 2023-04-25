import logging
import os

import torch

logger = logging.getLogger("train")

__all__ = [
    "mutate_sequence",
    "amino_distribution",
    "amino_char_to_index",
    "amino_alphabet",
    "encode_string_sequence",
    "encode_tensor_sequence",
]

amino_alphabet = [c for c in "ARNDCQEGHILKMFPSTWYVBZXJ*U"]
amino_char_to_index = {c: i for i, c in enumerate("ARNDCQEGHILKMFPSTWYVBZXJ*U")}

# the background distribution

# fmt: off
amino_frequencies = torch.tensor([0.074, 0.042, 0.044, 0.059, 0.033, 0.058, 0.037, 0.074,
                                  0.029, 0.038, 0.076, 0.072, 0.018, 0.040, 0.050, 0.081,
                                  0.062, 0.013, 0.033, 0.068])

amino_distribution = torch.distributions.categorical.Categorical(amino_frequencies)
# fmt: on

amino_n_to_v = torch.zeros(len(amino_char_to_index), 20)

for i in range(20):
    amino_n_to_v[i, i] = 1.0

amino_n_to_v[amino_char_to_index["B"], amino_char_to_index["D"]] = 0.5
amino_n_to_v[amino_char_to_index["B"], amino_char_to_index["N"]] = 0.5

amino_n_to_v[amino_char_to_index["Z"], amino_char_to_index["Q"]] = 0.5
amino_n_to_v[amino_char_to_index["Z"], amino_char_to_index["E"]] = 0.5

amino_n_to_v[amino_char_to_index["J"], amino_char_to_index["I"]] = 0.5
amino_n_to_v[amino_char_to_index["J"], amino_char_to_index["L"]] = 0.5

amino_n_to_v[amino_char_to_index["X"]] = amino_frequencies
amino_n_to_v[amino_char_to_index["*"]] = amino_frequencies
amino_n_to_v[amino_char_to_index["U"]] = amino_frequencies

# create vectors.
amino_a_to_v = {c: amino_n_to_v[i] for i, c in enumerate("ARNDCQEGHILKMFPSTWYVBZXJ*U")}


def encode_string_sequence(sequence: str) -> torch.Tensor:
    """Encode a string sequence as a tensor."""
    data = torch.zeros(20, len(sequence))
    for i, c in enumerate(sequence):
        # put a vector in.
        data[:, i] = amino_a_to_v[c]
    return data


def encode_tensor_sequence(sequence):
    """Encode a tensor of 1, 2, 3, etc as a fuzzy tensor."""
    data = torch.zeros(20, len(sequence))

    for i, c in enumerate(sequence):
        data[:, i] = amino_n_to_v[c]
    return data



def generate_string_sequence(length):
    sequence = amino_distribution.sample(sample_shape=(length,))
    sequence = "".join([amino_alphabet[i] for i in sequence])
    return sequence


def mutate_sequence(sequence, substitutions, sub_distributions):
    seq = sequence.clone()
    sub_indices = torch.randperm(len(seq))[:substitutions]

    for i in range(len(sub_indices)):
        # replace amino at position i
        # with the sampled amino from the substitution dist. at amino i
        try:
            # fmt: off
            seq[sub_indices[i]] = sub_distributions[amino_alphabet[seq[sub_indices[i]].item()]].sample()
            # fmt: on
        except KeyError as e:
            logger.debug(e)

    seq = seq.tolist()

    seq = torch.tensor(seq)

    return seq
