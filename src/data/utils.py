"""Utils file for working with training and evaluation sequence data"""
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from tqdm import tqdm

from src import models
from src.data.hmmerhits import FastaFile

HOME = os.environ["HOME"]


def update(d1, d2):
    c = d1.copy()
    for key in d2:
        if key in d1:
            c[key].update(d2[key])
        else:
            c[key] = d2[key]
    return c


def get_evaluation_data(
    query_id=4,
    save_dir=None,
    targethitsfile="/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargetdict.pkl",
    evaltargetfastafile="data/evaluationtargets.fa",
) -> Tuple[dict, dict, dict]:
    """Taking advantage of our current data structure of nested directories
    holding fasta files to quickly get all hmmer hits and sequence dicts for all
    queries in the input query id file and all target sequences in all of num_files"""

    query_id = str(query_id)

    queryfile = f"{HOME}/prefilter/uniref/split_subset/queries/queries_{query_id}.fa"
    queryfasta = FastaFile(queryfile)
    querysequences = queryfasta.data
    print(f"Number of query sequences: {len(querysequences)}")
    filtered_query_sequences = {}

    targetsequencefasta = FastaFile(evaltargetfastafile)
    targetsequences = targetsequencefasta.data
    print(f"Number of target sequences: {len(targetsequences)}")

    if save_dir is not None:  # only return those that we don't already have

        existing_queries = [f.strip(".txt") for f in os.listdir(save_dir)]

        print(f"Cleaning out {len(existing_queries)} queries that we already have in results...")
        for query, value in querysequences.items():
            if query not in existing_queries:
                filtered_query_sequences.update({query: value})

        querysequences = filtered_query_sequences

    with open(targethitsfile, "rb") as file:
        all_target_hits = pickle.load(file)

    print(f"Number of target HMMER hits: {len(all_target_hits)}")

    return querysequences, targetsequences, all_target_hits
