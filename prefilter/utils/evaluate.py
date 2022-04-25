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


def setup_hparams_and_model():
    parser = utils.create_parser()

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

    embeddings = torch.cat(embeddings, dim=0).contiguous()
    # create an index for easy searching
    index = utils.create_logo_index(embeddings, embed_dim, dev)
    # make labels into an array for easy np ops
    labels = np.asarray(labels)


def create_cluster_representative_index():
    return index, labels


# RealisticPairGenerator is a wrapper around Daniel's code
# How many "consensus" sequences to generate?
num_families = 1000
# self-explanatory

embeddings = []
labels = []
seed_sequences = []

afa_files = glob(
    "/home/tc229954/data/prefilter/pfam/seed/training_data/1000_file_subset/*-train.fa"
)
