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


def strobemer_representation(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    """Currently just experimenting with random matrix
    transformations and maxpooling"""

    maxpoollayer = nn.MaxPool2d(2)
    num_random_matrices = 10
    random_matrices = [
        torch.randint(1, 10, size=(embeddings[0].shape[1], embeddings[0].shape[1]))
        for i in range(num_random_matrices)
    ]

    outputs = []
    for embedding in embeddings:
        matrix_prod = torch.mm(embedding.float(), random.choice(random_matrices).float()).unsqueeze(
            0
        )
        output = maxpoollayer(matrix_prod)
        outputs.append(output)
    return outputs


def get_data_from_subset(
    dirpath: str = "/xdisk/twheeler/daphnedemekas/phmmer_max_results", query_id=0, file_num=1
) -> Tuple[dict, dict, dict]:
    """Taking advantage of our current data structure of nested directories
    holding fasta files to quickly get all hmmer hits and sequence dicts for all
    queries in the input query id file and all target sequences in all of num_files"""

    query_id = str(query_id)
    target = str(file_num)

    queryfile = f"{HOME}/prefilter/uniref/split_subset/queries/queries_{query_id}.fa"
    queryfasta = FastaFile(queryfile)
    hmmerhits = HmmerHits(dir_path=dirpath)

    # all_hits = {}
    querysequences = queryfasta.data
    targetsequences = {}

    print(f"uniref/split_subset/targets/targets_{target}.fa")

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{target}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    print(f"getting hits from target directory: {target} and query id {query_id}")

    target_hits = hmmerhits.get_hits(dirpath, target, query_num=query_id)
    for queryname, targethits in target_hits.items():
        for idx, targetname in enumerate(targethits):
            assert targetname in targetsequences, f"Target {idx} not in target sequences"
        assert queryname in querysequences

    print(
        f"Got {np.sum([len(target_hits[q]) for q in list(target_hits)])}\
             total hits from {dirpath}, target_id {file_num}, query_id {query_id}"
    )

    return querysequences, targetsequences, target_hits


def get_evaluation_data(
    dirpath: str = "/xdisk/twheeler/daphnedemekas/phmmer_max_results", query_id=4, val_targets=None
) -> Tuple[dict, dict, dict]:
    """Taking advantage of our current data structure of nested directories
    holding fasta files to quickly get all hmmer hits and sequence dicts for all
    queries in the input query id file and all target sequences in all of num_files"""

    query_id = str(query_id)

    queryfile = f"{HOME}/prefilter/uniref/split_subset/queries/queries_{query_id}.fa"
    queryfasta = FastaFile(queryfile)
    hmmerhits = HmmerHits(dir_path=dirpath)

    existing_queries = [
        f.strip(".txt")
        for f in os.listdir(
            "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities"
        )
    ]

    # all_hits = {}
    print("Cleaning out queries that we already have in results...")
    querysequences = queryfasta.data
    targetsequences = all_target_hits = filtered_query_sequences = {}

    for query, value in querysequences.items():
        if query not in existing_queries:
            filtered_query_sequences.update({query: value})

    querysequences = filtered_query_sequences
    if os.path.exists(f"{HOME}/prefilter/evaltargetsequences.pkl") and os.path.exists(
        f"{HOME}/prefilter/evaltargethmmerhits.pkl"
    ):
        print("Found existing evaluation data")
        with open("evaltargetsequences.pkl", "rb") as file:
            targetsequences = pickle.load(file)
        with open("evaltargethmmerhits.pkl", "rb") as file:
            all_target_hits = pickle.load(file)
        return querysequences, targetsequences, all_target_hits

    for tfile in tqdm(range(45)):

        targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{tfile}.fa")
        targetdata = targetfasta.data
        eval_targets = {}

        for target, values in targetdata.items():
            if target in val_targets:
                eval_targets.update({target: values})
        targetsequences.update(eval_targets)
        target_hits = hmmerhits.get_hits(
            dirpath, tfile, query_num=query_id, filtered_targets=list(eval_targets)
        )  # {'target_dirnum' :{'query_dirnum': {qname: {tname: data} }  } }

        all_target_hits.update(target_hits)
        print("Saving target to a file...")
    with open("evaltargetsequences.pkl", "wb") as evaltargetfile:
        pickle.dump(targetsequences, evaltargetfile)
    print("Saving hits to a file...")
    with open("evaltargethmmerhits.pkl", "wb") as evalhitsfile:
        pickle.dump(all_target_hits, evalhitsfile)

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
        checkpoint_path=checkpoint_path,
        map_location=device,
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


def get_actmaps(embeddings: list, function="sum", exp=2, show=False, title=None):
    """Calculate activation maps based on the given function
    on the embedding vectors"""
    num_embeds = 500
    figname1 = None
    if len(embeddings) > num_embeds:
        embeddings = random.sample(embeddings, num_embeds)

    else:
        print(f"Have {len(embeddings)} embeddings ")
        num_embeds = len(embeddings)

    seq_dim = np.max([len(s) for s in embeddings])

    all_actmaps = np.zeros((num_embeds, seq_dim, 1, 3))

    # embeddings shape (num_embeddings, seq_len, 256)
    for idx in range(num_embeds):
        sample = embeddings[idx]
        if function == "sum":
            outputs = (sample**exp).sum(1)
        elif function == "max":
            outputs = (sample**exp).max(1)
        try:
            outputs_n = outputs.reshape(1, outputs.shape[0])
            outputs_n = outputs_n / outputs_n.sum(axis=1)
        except AttributeError as error:
            print(error)
        amp = outputs_n[0, ...]
        amp = amp.numpy()
        amp = 255 * (amp - np.min(amp)) / (np.max(amp) - np.min(amp) + 1e-12)
        amp = np.uint8(np.floor(amp))
        amp = cv2.applyColorMap(amp, cv2.COLORMAP_JET)
        seqlen = amp.shape[0]
        all_actmaps[idx, :seqlen, :, :] = amp
        np.save(figname1, all_actmaps, allow_pickle=True)

    if show:
        figname1 = f"activationmaps_{np.random.randint(0,5000)}.png"

        all_actmaps = all_actmaps.reshape(num_embeds, seq_dim, 3)
        plt.imshow(all_actmaps, aspect=4)
        if title:
            plt.title(title)

        print(f"Saving figure as {figname1}")
        plt.savefig(figname1)

    return all_actmaps


def actmap_pipeline(names: List[str], embeddings: List[torch.Tensor], max_hmmer_hits: dict):
    """Pipeline to calculate actmaps given embedding vectors"""

    pos_samples, neg_samples = get_subsets(max_hmmer_hits)  # names of sequences

    similar_embeddings = []

    diff_embeddings = []

    for seq_name in pos_samples:
        try:
            idx = names.index(seq_name)
            emb = embeddings[idx]

            similar_embeddings.append(emb)
        except ValueError as error:
            print(error)

    print(f"Got {len(similar_embeddings)}  similar embeddings")

    for seq_name in neg_samples:
        idx = names.index(seq_name)
        emb = embeddings[idx]
        diff_embeddings.append(emb)

    print(f"Got {len(diff_embeddings)}  dissimilar embeddings")

    get_actmaps(embeddings, title="Amino activation maps")
    get_actmaps(similar_embeddings, title="Similar embedding activation map")
    get_actmaps(diff_embeddings, title="Different embedding activation map")
