import os
import numpy as np
import pdb
import random
import matplotlib.pyplot as plt
from typing import Tuple

import pytorch_lightning as pl
from tqdm import tqdm
from typing import List
import torch
import torch.nn as nn

from src.data.hmmerhits import HmmerHits, FastaFile

from src.utils import pluginloader, encode_string_sequence
from src import models

HOME = os.environ["HOME"]


def strobemer_representation(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    """Currently just experimenting with random matrix
    transformations and maxpooling"""

    """
    Literally nn.linear except we don't train it
    hashes are going to be random affine transformations 
    the mean and std should be 0 and 1, nice gaussian
    pretty sure that's how nn linear weights off the bat 
    a bunch of random nn linear untrained functions 

    taking our kmer features and mapping it to magnitude 

    to start, we will do encoding on one amino at at ime 
    then optimize the 'amino hashes'
    we get a new vector
    - pseudo random vectors, random affine transformation m(x+b)
    - the bias means every big factor has a chance to be large
    every small vector has the chance to be large still too

    then we do strobemer optimisation 
    instead of doing a DP matrix to find minimum/maximum sum we do a preprogrammed 
    convolution of these things

    actual max pooling??
    kernal takes specific combinations of our space 

    what is the DP thing?

    the kernel calculation is an identity matrix 
    on the sequence dimension of the kernel everything is 0 except our aminos that we are interested in optimizing 

    doing the convolution on (seq length x embedding dimension) to choose a subset of the embedding vectors 

    the aminos already have information about the surrounding context 
    we want to figure out what amino features are gonna be the most consistent 

    good INVARIANT set of features 

    map our nice vector space into a random vector space to approximate the hashes
    a minimum subset of kmers, or we can randomly approximate that 
    i think because our vectors are more descriptive than random kmers, approximating is ok 
    approximating = randomly sampling a subset of vectors 
    exact = normal DP indexing 
    do a DP matrix type thing to find a minimum hash in strobemers, we do the same thing only instead of minimizing hash values
    we are minimizing dot products between amino vectors inside a window (dot product of 4 vectors in kmer)

    SEQUENCE -> CNN -> RANDOM TRANSFORMATION M(X+B) (linear layer without bias (X+B) WHERE B IS A RANDOMLY INITIALIZED GAUSSIAN
    AND X IS THE VECTOR OF AMINO ACID EMBEDDINGS 
    -- do different versions on the same aminos --end up with a bigger space than i started with
    -> once we have these we do our strobemer optimization to turn WINDOWS of sequence into a single vector (DOT PRODUCT / ADDING)
    THEN
    we have one vector to represent an area. 

    if you want you can do convolution rather than M (M = nn linear)

    you want to make sure that B and M are initialized as random gaussians 
    (for the bias the std should be the std of the magnitude of all your vectors in your training set )

    InstanceNorm to nromalise our embeddings ???
        - to replace the +b part 
    

    normalising and dot producting makes us think of that paper we saw in ML

    ok:: instead of Mx+B we do M(N(X)) marix multiplication times some normalisation of X

    
    now we have representation of one window of sequence : at the end of this reduction we have 4 representations of the sequence 
    length 64 sequence getting reduced to 4 vectors 

    then we feed that into FAISS 
    but for now we do all versus all dot product on our reduced spaces 

    that number (4) is going to depend on the sequence length
        - take dot prods 
        - take the minimum 
        - window length W
        - windows have overlap 
        - windows including overlap length = 32
        - 64 length sequence 
        - window size 32 with 8 overlap we have compression factor of 16 
        64 / 16 = 4
        (sequence / compression factor) (compression factor = W - overlap for window, = how many windows represent that sequence)
    
    if you are using multiple hash functino you use whichever optimization is the best
    - whichever gives you smaller / larger dot products

    our assumption is the best hash function for two homologous sequences will be the same hash function

    a bunch of places for optimisation in vector searching
    GPU level optimizations?? that would be awesome --TIM

    """

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
    t = str(file_num)

    queryfile = f"{HOME}/prefilter/uniref/split_subset/queries/queries_{query_id}.fa"
    queryfasta = FastaFile(queryfile)
    hmmerhits = HmmerHits(dir_path=dirpath)

    # all_hits = {}
    querysequences = queryfasta.data
    targetsequences = {}

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{t}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    print(f"getting hits from target directory: {t} and query id {query_id}")

    target_hits = hmmerhits.get_hits(
        dirpath, t, query_num=query_id
    )  # {'target_dirnum' :{'query_dirnum': {qname: {tname: data} }  } }

    # qnames = list(target_query_hits[t][query_id].keys())

    for queryname, targethits in target_hits.items():
        for idx, targetname in enumerate(targethits.keys()):
            assert targetname in targetsequences.keys(), f"Target {idx} not in target sequences"
        assert queryname in querysequences.keys()

    print(
        f"Got {np.sum([len(target_hits[q]) for q in list(target_hits.keys())])} total hits from {dirpath}, target_id {file_num}, query_id {query_id}"
    )

    return querysequences, targetsequences, target_hits


# LOCALLY -- make some distributions of e values?


def get_embeddings(targetsequences: dict, querysequences: dict):
    """Method to get embeddings from sequences
    without having to declare a class"""
    from src.utils import pluginloader, encode_string_sequence

    model_dict = {
        m.__name__: m for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
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


def get_actmaps(embeddings: list, function="sum", p=2, show=False, title=None):
    import cv2

    """Calculate activation maps based on the given function
    on the embedding vectors"""
    num_embeds = 500
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
            outputs = (sample**p).sum(1)
        elif function == "max":
            outputs = (sample**p).max(1)
        try:
            outputs_n = outputs.reshape(1, outputs.shape[0])
            outputs_n = outputs_n / outputs_n.sum(axis=1)
        except AttributeError:
            pdb.set_trace()
        am = outputs_n[0, ...]
        am = am.numpy()
        am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
        am = np.uint8(np.floor(am))
        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
        seqlen = am.shape[0]
        all_actmaps[idx, :seqlen, :, :] = am
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
        except ValueError as e:
            print(e)

    print(f"Got {len(similar_embeddings)}  similar embeddings")

    for seq_name in neg_samples:
        idx = names.index(seq_name)
        emb = embeddings[idx]
        diff_embeddings.append(emb)

    print(f"Got {len(diff_embeddings)}  dissimilar embeddings")

    get_actmaps(embeddings, title="Amino activation maps")
    get_actmaps(similar_embeddings, title="Similar embedding activation map")
    get_actmaps(diff_embeddings, title="Different embedding activation map")
