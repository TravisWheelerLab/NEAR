"""Utils file for working with training and evaluation sequence data"""
import os
import pickle
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from tqdm import tqdm

from src import models
from src.data.hmmerhits import FastaFile, HmmerHits

HOME = os.environ["HOME"]


def update(d1, d2):
    c = d1.copy()
    for key in d2:
        if key in d1:
            c[key].update(d2[key])
        else:
            c[key] = d2[key]
    return c


def get_evaluation_data_old(
    hitsdirpath=None,
    query_id=4,
    save_dir=None,
    targethitsfile="evaltargethmmerhits.pkl",
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


# LOCALLY -- make some distributions of e values?


def get_embeddings(targetsequences: dict, querysequences: dict):
    """Method to get embeddings from sequences
    without having to declare a class"""
    from src.utils import encode_string_sequence, pluginloader

    model_dict = {
        m.__name__: m for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
    }

    model_class = model_dict["ResNet1d"]
    checkpoint_path = f"{HOME}/prefilter/ResNet1d/4/checkpoints/best_loss_model.ckpt"
    device = "cuda"

    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=device,
    ).to(device)
    print("Loaded model")

    target_embeddings = {}
    if targetsequences is not None:
        print("get target embeddings...")
        for tname, sequence in tqdm(targetsequences.items()):
            embedding = (
                model(encode_string_sequence(sequence).unsqueeze(0).to(device)).squeeze().T  # 400
            )  # [400, 256]

            target_embeddings[tname] = embedding.cpu()

    query_embeddings = {}

    if querysequences is not None:
        print("get query embeddings...")

        for tname, sequence in tqdm(list(querysequences.items()[:10])):
            embedding = (
                model(encode_string_sequence(sequence).unsqueeze(0).to(device)).squeeze().T  # 400
            )  # [400, 256]

            query_embeddings[tname] = embedding.cpu()
        print("got embeddings")
    return target_embeddings, query_embeddings


def get_subsets(hits_data: dict, score_threshold_high=100, score_threshold_low=3):
    """Get subsets of the hits data that are less than the low threshold
    and greater than the high threshold"""
    pos_samples = []
    neg_samples = []
    max_hits = 0
    for query in hits_data:
        num_hits = len(hits_data[query])
        if num_hits > max_hits:
            max_hits = num_hits
            query_name = query

    target_data = hits_data[query_name]
    print(f"There are {len(target_data)} entries in the hits for this query")
    for tname, data in target_data.items():
        if data[1] > score_threshold_high:
            pos_samples.append(tname)
        elif data[1] < score_threshold_low:
            neg_samples.append(tname)
    return pos_samples, neg_samples
