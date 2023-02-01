from src.data.hmmerhits import HmmerHits, FastaFile
import os
import numpy as np
import pdb

HOME = os.environ["HOME"]
from src.utils import pluginloader
from src import models
import torch
from src.utils import encode_string_sequence
import random
import matplotlib.pyplot as plt
import cv2
from typing import Tuple
import pytorch_lightning as pl
from tqdm import tqdm


def get_data_from_subset(
    dirpath: str = "uniref/phmmer_results", query_id: str = "0", num_files: int = 5
) -> Tuple[dict, dict, dict]:
    """Taking advantage of our current data structure of nested directories
    holding fasta files to quickly get all hmmer hits and sequence dicts for all
    queries in the input query id file and all target sequences in all of num_files"""

    queryfile = f"uniref/split_subset/queries/queries_{query_id}.fa"
    queryfasta = FastaFile(queryfile)
    hmmerhits = HmmerHits(dir_path=dirpath)

    all_hits = {}
    querysequences = queryfasta.data
    targetsequences = {}

    for t in range(num_files):
        t = str(t)
        targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{t}.fa")
        targetdata = targetfasta.data
        targetsequences.update(targetdata)

        target_dir = os.path.join(hmmerhits.dir_path, t)

        target_query_hits, _ = hmmerhits.get_hits(
            target_dir, query_num=query_id
        )  # {'target_dirnum' :{'query_dirnum': {qname: {tname: data} }  } }

        qnames = list(target_query_hits[t][query_id].keys())

        target_hits = target_query_hits[t][
            query_id
        ]  # {queryname: {targetname : [data]}}
        for qname in qnames:
            if qname in all_hits.keys():
                qdict = all_hits[qname]
                qdict.update(target_hits[qname])
                all_hits[qname] = qdict
            else:
                all_hits.update({qname: target_hits[qname]})

    print(
        f"Got {np.sum([len(all_hits[q]) for q in list(all_hits.keys())])} total hits from {dirpath}"
    )

    return querysequences, targetsequences, all_hits


# LOCALLY -- make some distributions of e values?


def get_embeddings(targetsequences: dict, querysequences: dict):
    model_dict = {
        m.__name__: m
        for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
    }

    model_class = model_dict["ResNet1d"]
    checkpoint_path = f"{HOME}/prefilter/ResNet1d/4/checkpoints/best_loss_model.ckpt"
    device = "cuda"

    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device(device),
    ).to(device)
    print("Loaded model")

    target_embeddings = {}
    print("get target embeddings...")
    for tname, sequence in tqdm(targetsequences.items()):
        embedding = (
            model(encode_string_sequence(sequence).unsqueeze(0).to(device))  # 400
            .squeeze()
            .T
        )  # [400, 256]

        target_embeddings[tname] = embedding.cpu().detach().numpy()

    query_embeddings = {}

    if querysequences is not None:
        print("get query embeddings...")

        for tname, sequence in tqdm(querysequences.items()):
            embedding = (
                model(encode_string_sequence(sequence).unsqueeze(0).to(device))  # 400
                .squeeze()
                .T
            )  # [400, 256]

            query_embeddings[tname] = embedding.cpu().detach().numpy()
        print("got embeddings")
    return target_embeddings, query_embeddings


# activation maps


def get_actmaps(embeddings: list, function="sum", p=2, show=True, title=None):
    num_embeds = 500
    if len(embeddings) > num_embeds:
        embeddings = random.sample(embeddings, num_embeds)

    else:
        print(f"Have {len(embeddings)} embeddings ")
        num_embeds = len(embeddings)

    seq_dim = np.max([len(s) for s in embeddings])

    all_actmaps = np.zeros((num_embeds, seq_dim, 1, 3))

    # embeddings shape (num_embeddings, seq_len, 256)

    for idx, sample in enumerate(embeddings):
        if function == "sum":
            outputs = (sample**p).sum(1)
        elif function == "max":
            outputs = (sample**p).max(1)
        outputs_n = outputs.reshape(1, outputs.shape[0])
        outputs_n = outputs_n / outputs_n.sum(axis=1)
        am = outputs_n[0, ...]
        am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
        am = np.uint8(np.floor(am))
        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        seqlen = am.shape[0]
        all_actmaps[idx, :seqlen, :, :] = am
    if show:
        all_actmaps = all_actmaps.reshape(seq_dim, num_embeds, 3)
        plt.imshow(all_actmaps, aspect=0.2)
        if title:
            plt.title(title)

        figname = f"activationmaps_{np.random.randint(0,5000)}.png"
        print(f"Saving figure as {figname}")
        plt.savefig(figname)
        # sliced_actmaps = all_actmaps[:,:512, :]
        # plt.clf()
        # plt.imshow(sliced_actmaps, aspect = 3)
        # if title:
        #     plt.title(title + ', sliced to 512')

        # figname = f'activationmaps_sliced_{np.random.randint(0,5000)}.png'
        # print(f'Saving figure as {figname}')
        # plt.savefig(figname)
    return all_actmaps


#

# do the above but when we have small e evalues


def get_subsets(hits_data: dict, score_threshold_high=100, score_threshold_low=3):
    pos_samples = []
    neg_samples = []
    max_hits = 0
    for q in hits_data.keys():
        num_hits = len(hits_data[q])
        if num_hits > max_hits:
            max_hits = num_hits
            query_name = q

    target_data = hits_data[query_name]
    print(f"There are {len(target_data)} entries in the hits for this query")
    for tname, data in target_data.items():
        if data[1] > score_threshold_high:
            pos_samples.append(tname)
        elif data[1] < score_threshold_low:
            neg_samples.append(tname)
    return pos_samples, neg_samples


def actmap_pipeline():
    _, targetsequences, all_hits_normal = get_data_from_subset(
        "uniref/phmmer_normal_results", 1
    )

    target_embeddings, _ = get_embeddings(targetsequences, None)

    embeddings = list(target_embeddings.values())

    # pos_samples, neg_samples = get_subsets(all_hits_normal)

    # similar_embeddings = []

    # diff_embeddings = []

    # for pair in pos_samples:
    #     seq = target_embeddings[pair]

    #     similar_embeddings.append(seq)

    # print(f'Got {len(pos_samples)}  similar embeddings')

    # for pair in neg_samples:
    #     seq = target_embeddings[pair]

    #     diff_embeddings.append(seq)

    # print(f'Got {len(neg_samples)}  dissimilar embeddings')
    actmaps = []
    for i in range(5):
        key = list(target_embeddings.keys())[i]
        print(key)
        print(targetsequences[key])
        # actmap = get_actmaps([embeddings[i]], function = 'max',title = f'Amino {i} activation map')
        outputs = (embeddings[i] ** 2).max(1)

        outputs_n = outputs.reshape(1, outputs.shape[0])
        outputs_n = outputs_n / outputs_n.sum(axis=1)
        am = outputs_n[0, ...]
        am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
        am = np.uint8(np.floor(am))
        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        actmaps.append(am)
    seq1 = "MAKPVPPPRPVQPPLSRTAHWRRAHRMTILLLLLWLFTGFGAAFFARELAGLSVFGWPLSFYLAAQGASLVYLGIIVFYAWRMRQLDRAAAQATVPEPHA"
    seq2 = "MKKGCKIALVGGLAVVVGAVACWKWMIRMPLGEFTRCALYMAVMDDEICRNELEGNRIGDEVITFPPKEESLQYRYHLFLQMNLRKTSAQLQEEIADMEQRLQESRQYIGQPKEQLEVEFVEQGAE"

    fig = plt.figure(figsize=(20, 100))
    plt.imshow(actmaps[0], aspect=0.2)
    plt.yticks(range(100), labels=seq1, fontsize=60, rotation=90)
    plt.savefig("0.png")

    fig = plt.figure(figsize=(20, 120))
    plt.imshow(actmaps[1], aspect=0.2)
    plt.yticks(range(126), labels=seq2, fontsize=60, rotation=90)
    plt.savefig("1.png")
    pdb.set_trace()
    # get_actmaps(similar_embeddings, function = 'max', title = 'Similar embedding activation map')
    # get_actmaps(diff_embeddings,  function = 'max', title = 'Different embedding activation map')
