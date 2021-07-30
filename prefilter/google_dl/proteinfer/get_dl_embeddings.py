# Commented out IPython magic to ensure Python compatibility.
# !git clone https://github.com/google-research/proteinfer
# %cd proteinfer
# !pip3 install -qr  requirements.txt

import os
from glob import glob

import numpy as np

import inference


def fasta_from_file(fasta_file):
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []

    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        sequence_strs.append("".join(buf))
        cur_seq_label = None
        buf = []

    with open(fasta_file, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:  # sequence line
                buf.append(line.strip())

    _flush_current_seq()

    # print(len(set(sequence_labels)), len(sequence_labels))
    # assert len(set(sequence_labels)) == len(sequence_labels)

    return sequence_labels, sequence_strs


# Get a savedmodel
# !wget -qN https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/models/single_domain_per_sequence_zipped_models/seed_random_32.0/5356760.tar.gz
# unzip
# !tar xzf 5356760.tar.gz
# Get the vocabulary for the savedmodel, which tells you which output index means which family
# !wget https://storage.googleapis.com/brain-genomics-public/research/proteins/pfam/models/single_domain_per_sequence_zipped_models/trained_model_pfam_32.0_vocab.json

# Find the unzipped path
# !ls *5356760*

# Load savedmodel

def get_embeddings_from_sequences_and_labels(sequences, out_dir):
    for f in sequences:
        sequence_labels, sequence_strs = fasta_from_file(f)
        # print(sequence_labels, sequence_strs)

        activations = inferrer.get_activations(sequence_strs)
        # Find what the most likely class is
        for sequence_label, sequence_embedding in zip(sequence_labels,
                                                      activations):
            # print(sequence_label, sequence_embedding.shape)
            of = os.path.join(out_dir, sequence_label + 'npy')
            np.save(of, sequence_embedding)


if __name__ == '__main__':
    root = '../../small-dataset/'
    test_sequences = glob(os.path.join(root, 'fasta/*test*'))
    train_sequences = glob(os.path.join(root, 'train_subset/*'))
    consensus_sequences = glob(os.path.join('../../', 'consensus_sequences/*PF*'))

    # print(len(test_sequences), len(train_sequences), len(consensus_sequences))

    inferrer = inference.Inferrer(
        'trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760',
        use_tqdm=False,
        batch_size=1,
        activation_type="pooled_representation"
    )

    # print(consensus_sequences)
    # get_embeddings_from_sequences_and_labels(test_sequences,
    #         './embeddings/test/')
    # get_embeddings_from_sequences_and_labels(train_sequences,
    #         './embeddings/train/')
    get_embeddings_from_sequences_and_labels(consensus_sequences,
                                             './embeddings/consensus/')
