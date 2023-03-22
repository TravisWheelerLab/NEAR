import os

import torch

HOME = os.environ["HOME"]
import logging
import pdb
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl

from data.utils import get_data_from_subset
from src import models
from src.evaluators.contrastive import ContrastiveEvaluator
from src.utils import encode_string_sequence, pluginloader

logger = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO)

logger.setLevel(logging.WARNING)
from collections import defaultdict

log_verbosity = logging.INFO
num_threads = 12

normalize_embeddings = True
index_string = "Flat"
nprobe = 1
unrolled_names = []
comp_func = np.greater_equal
distance_threshold = 0
encoding_func = None
model_device = "cuda"
index_device = "cuda"
figure_path = f"{HOME}/roc_test.png"
minimum_seq_length = 0
max_seq_length = 10000

model_dict = {
    m.__name__: m
    for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
}

model_class = model_dict["ResNet1d"]
checkpoint_path = (
    f"{HOME}/prefilter/ResNet1d/4/checkpoints/best_loss_model.ckpt"
)
device = "cuda"

model = model_class.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    map_location=torch.device(device),
).to(device)
logger.info("Loaded model")

querysequences, targetsequences, all_hits = get_data_from_subset(
    "uniref/phmmer_results", num_files=2
)

evaluator = ContrastiveEvaluator(
    query_seqs=querysequences,
    target_seqs=targetsequences,
    hmmer_hits_max=all_hits,
    encoding_func=encoding_func,
    model_device=model_device,
    index_device=index_device,
    figure_path=figure_path,
    minimum_seq_length=minimum_seq_length,
    max_seq_length=max_seq_length,
)

result = evaluator.evaluate(model_class=model)
pdb.set_trace()
raise


# which ones do we filter out compared to HMMER's prefilter?
# would be better if you changed the arguments that the dataloader / eval config take so we can feed them sequence dicts directly
# then can use the classes


# also wnat to make some correlation plots between the HMMER hits data and our index data? whatever that is


raise
qname2 = list(target_query_hits["0"]["0"].keys())[1]
queryseq2 = queryfasta.data[qname2]

target_hits2 = target_query_hits["0"]["0"][qname2]


testset = {}

sequences = [queryseq1, queryseq2]
scores = []
e_values = []
for tname, data in target_hits1.items():

    targetseq = targetfasta.data[tname]

    sequences.append(targetseq)
    scores.append(data[1])
    e_values.append(data[0])

for tname, data in target_hits2.items():

    targetseq = targetfasta.data[tname]

    sequences.append(targetseq)
    scores.append(data[1])
    e_values.append(data[0])


model_dict = {
    m.__name__: m
    for m in pluginloader.load_plugin_classes(models, pl.LightningModule)
}

model_class = model_dict["ResNet1d"]
checkpoint_path = (
    f"{HOME}/prefilter/ResNet1d/4/checkpoints/best_loss_model.ckpt"
)
device = "cuda"

model = model_class.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    map_location=torch.device(device),
).to(device)

embeddings = []
for sequence in sequences:
    embedding = (
        model(encode_string_sequence(sequence).unsqueeze(0).to(device))
        .squeeze()
        .T  # 400
    )  # [400, 256]

    embeddings.append(embedding)

MAX = sys.maxsize
from collections import defaultdict, deque


def thinner(hash_list, w):
    """
    Input: a list with hash values
    Output: A list with tuples: (pos in original list, minimim hash value) for each window of w hashes
    """
    window_hashes = deque(hash_list[:w])
    min_index, curr_min_hash = argmin(window_hashes)
    thinned_hash_list = [(min_index, curr_min_hash)]

    for i in range(w, len(hash_list) + w - 1):
        if i >= len(hash_list):
            new_hash = MAX
        else:
            new_hash = hash_list[i]
        # updating window
        discarded_hash = window_hashes.popleft()
        window_hashes.append(new_hash)

        # we have discarded previous windows minimizer, look for new minimizer brute force
        if curr_min_hash == discarded_hash:
            min_index, curr_min_hash = argmin(window_hashes)
            thinned_hash_list.append((min_index + i + 1 - w, curr_min_hash))

        # Previous minimizer still in window, we only need to compare with the recently added kmer
        elif new_hash < curr_min_hash:
            curr_min_hash = new_hash
            thinned_hash_list.append((i, curr_min_hash))

    return


def argmin(array):
    min_index = array.index(min(array))
    min_val = array[min_index]
    return min_index, min_val


def randstrobe_order2(hash_seq_list, start, stop, hash_m1, prime):
    min_index, min_value = argmin(
        [(hash_m1 + hash_seq_list[i][1]) % prime for i in range(start, stop)]
    )
    min_hash_val = hash_m1 - hash_seq_list[start + min_index][1]
    return min_index, min_hash_val


def seq_to_randstrobes2_iter(
    seq, k_size, strobe_w_min_offset, strobe_w_max_offset, prime, w
):
    # print([seq[i:i+k_size].mean().round(2) for i in range(len(seq) - k_size +1)][:10])
    hash_seq_list = [
        (i, hash(seq[i : i + k_size].sum()))
        for i in range(len(seq) - k_size + 1)
    ]
    # print(hash_seq_list[:10])
    if w > 1:
        hash_seq_list_thinned = thinner(
            [h for i, h in hash_seq_list], w
        )  # produce a subset of positions, still with samme index as in full sequence
    else:
        hash_seq_list_thinned = hash_seq_list

    # assert len(hash_seq_list[:-k_size]) == len(hash_seq_list) - k_size

    for (p, hash_m1) in hash_seq_list_thinned:  # [:-k_size]:
        if p >= len(hash_seq_list) - k_size:
            break
        # hash_m1 = hash_seq_list[p]
        window_p_start = (
            p + strobe_w_min_offset
            if p + strobe_w_max_offset <= len(hash_seq_list)
            else max(
                (p + strobe_w_min_offset)
                - (p + strobe_w_max_offset - len(hash_seq_list)),
                p,
            )
        )
        window_p_end = min(p + strobe_w_max_offset, len(hash_seq_list))
        # print(window_p_start, window_p_end)
        min_index, hash_value = randstrobe_order2(
            hash_seq_list, window_p_start, window_p_end, hash_m1, prime
        )
        p2 = window_p_start + min_index
        yield p, p2, hash_value


def randstrobes(
    seq, k_size, strobe_w_min_offset, strobe_w_max_offset, w, order=2
):
    prime = 997
    assert (
        strobe_w_min_offset > 0
    ), "Minimum strobemer offset has to be greater than 0 in this implementation"
    if order == 2:
        if k_size % 2 != 0:
            print(
                "WARNING: kmer size is not evenly divisible with 2, will use {0} as kmer size: ".format(
                    k_size - k_size % 2
                )
            )
            k_size = k_size - k_size % 2
        m_size = k_size // 2
        randstrobes = {
            (p1, p2): h
            for p1, p2, h in seq_to_randstrobes2_iter(
                seq, m_size, strobe_w_min_offset, strobe_w_max_offset, prime, w
            )
        }
        return randstrobes


numpy_embeddings = []
rounded_embeddings = []
for emb in embeddings:
    array = emb.cpu().detach().numpy()
    numpy_embeddings.append(array)
    rounded_embeddings.append(array.round(3))

num_embeddings = len(embeddings)

cossims = []

# TEST IF THEY HAVE CORRELATION IN THE COSIN SIMILAIRTY


KSIZE = 4
strobe_w_min_offset = 3
strobe_w_max_offset = 10
w = 0
rs_per_seq = []
for seq_emb in rounded_embeddings:
    amino_embeddings = []
    for amino_emb in seq_emb:
        amino_emb = (amino_emb - np.min(amino_emb)) / (
            np.max(amino_emb) - np.min(amino_emb)
        )
        r = randstrobes(
            amino_emb, KSIZE, strobe_w_min_offset, strobe_w_max_offset, w
        )
        amino_embeddings += list(r.values())
    rs_per_seq.append(amino_embeddings)

shingles = [set(r) for r in rs_per_seq]

num_elements = int(num_embeddings * (num_embeddings - 1) / 2)


query1 = shingles[0]

query2 = shingles[1]

matches1 = []
matches2 = []

for t in shingles[2 : 2 + len(target_hits1)]:
    matches1.append(len(shingles[0] & t))
for t in shingles[2 + len(target_hits1) :]:
    matches2.append(len(shingles[1] & t))


pdb.set_trace()
JSim = [0 for x in range(num_elements)]
estJSim = [0 for x in range(num_elements)]


import sys


# Define a function to map a 2D matrix coordinate into a 1D index.
def getTriangleIndex(i, j):
    # If i == j that's an error.
    if i == j:
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    # If j < i just swap the values.
    if j < i:
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array.
    # This fancy indexing scheme is taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # But I adapted it for a 0-based index.
    # Note: The division by two should not truncate, it
    #       needs to be a float.
    k = int(i * (num_embeddings - (i + 1) / 2.0) + j - i) - 1

    return k


for i in range(0, num_embeddings):

    # Retrieve the set of shingles for document i.
    s1 = shingles[i]

    for j in range(i + 1, num_embeddings):
        s2 = shingles[j]

        # Calculate and store the actual Jaccard similarity.
        JSim[getTriangleIndex(i, j)] = len(s1.intersection(s2)) / len(
            s1.union(s2)
        )

maxShingleID = max([max(e) for e in shingles])
numHashes = 10


def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []

    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID)

        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID)

        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1

    return randList


# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

# List of documents represented as signature vectors
signatures = []
nextPrime = 4294967311
# Rather than generating a random permutation of all possible shingles,
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index
# of the first shingle that you would have encountered in the random order.

# For each embedding...
for emb in range(num_embeddings):

    # Get the shingle set for this embedding.
    shingleIDSet = shingles[emb]

    # The resulting minhash signature for this embedding.
    signature = []

    # For each of the random hash functions...
    for i in range(0, numHashes):

        # For each of the shingles actually in the document, calculate its hash code
        # using hash function 'i'.

        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
        # the maximum possible value output by the hash.
        minHashCode = nextPrime + 1

        # For each shingle in the document...
        for shingleID in shingleIDSet:
            # Evaluate the hash function.
            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

            # Track the lowest hash code seen.
            if hashCode < minHashCode:
                minHashCode = hashCode

        # Add the smallest hash code value as component number 'i' of the signature.
        signature.append(minHashCode)

    # Store the MinHash signature for this document.
    signatures.append(signature)

# For each of the test documents...
for i in range(0, num_embeddings):
    # Get the MinHash signature for document i.
    signature1 = signatures[i]

    # For each of the other test documents...
    for j in range(i + 1, num_embeddings):

        # Get the MinHash signature for document j.
        signature2 = signatures[j]

        count = 0
        # Count the number of positions in the minhash signature which are equal.
        for k in range(0, numHashes):
            count = count + (signature1[k] == signature2[k])

        # Record the percentage of positions which matched.
        estJSim[getTriangleIndex(i, j)] = count / numHashes

pdb.set_trace()
