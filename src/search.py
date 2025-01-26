#!/usr/bin/env python3

import argparse
import faiss
import pickle
import sys
import numpy as np
import process_hits
from Bio import SeqIO
from typing import NamedTuple, Optional, TextIO
import time

class Args(NamedTuple):
    """Command-line arguments"""

    query_embeddings_path: str
    target_embeddings_path: str

    query_fasta_path: str
    target_fasta_path: str

    output_path: str

    score_adjustment: float

    index_str: str

    gpu: bool


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    # Usage: faiss_search.py <query path:softmasked fasta> <target
    # path:softmasked fasta> <output path> <gpu|cpu> [embedding start, end]")

    parser = argparse.ArgumentParser(
        description="""Embeddings files are expected to be np arrays
        of shape [n,d] where n is the number of sequences,
        and d is the embedding dimensionality""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-q",
        "--query",
        help="Query embedding file",
        metavar="FILE",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-t",
        "--target",
        help="Target embedding file",
        metavar="FILE",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--query_fasta",
        help="Query softmasked FASTA file",
        metavar="FILE",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--target_fasta",
        help="Target softmasked FASTA file",
        metavar="FILE",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-s",
        "--score_adjustment",
        help="Noise gating adjustment to scores before accumulation",
        type=float,
        default=3.0)

    parser.add_argument(
        "--index",
        help="FAISS index string",
        type=str,
        default="IVF5000,PQ32")

    parser.add_argument(
        "-o",
        "--output",
        help="Output file",
        metavar="FILE",
        type=str,
        default="out.txt",
    )

    parser.add_argument("-g", "--gpu", help="GPU", action="store_true")

    args = parser.parse_args()


    return Args(
        query_embeddings_path=args.query,
        target_embeddings_path=args.target,
        query_fasta_path=args.query_fasta,
        target_fasta_path=args.target_fasta,
        output_path=args.output,
        index_str=args.index,
        gpu=args.gpu,
        score_adjustment=args.score_adjustment
    )


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""
    global_start = time.time()

    args = get_args()
    query_path = args.query_embeddings_path
    softmask_query_path = args.query_fasta_path
    target_path = args.target_embeddings_path
    softmask_target_path = args.target_fasta_path
    index_str = args.index_str
    score_adjustment = args.score_adjustment

    print(f"Reading query file: {query_path}")
    query_data = read_embeddings(
        query_path,
        softmask_query_path
    )

    print(f"Reading target file: {target_path}")
    target_data = read_embeddings(
        target_path,
        softmask_target_path,
    )

    print("Preparing query data")
    query_seq_ids, query_embeddings, query_starts = prepare_embedding_data(
        query_data
    )

    print("Preparing target data")
    target_seq_ids, target_embeddings, target_starts = (
        prepare_embedding_data(target_data)
    )
    #print(query_starts[0:10])
    print(query_embeddings.shape)
    print(target_embeddings.shape)

    print("Creating index from target data")
    start_time = time.time()

    score_adjustment = score_adjustment / target_embeddings.shape[-1]**0.5
    target_index = faiss.index_factory(
        target_embeddings.shape[-1], index_str, faiss.METRIC_INNER_PRODUCT
    )

    if args.gpu:
        gpu_resource = faiss.StandardGpuResources()
        target_index = faiss.index_cpu_to_gpu(gpu_resource, 0, target_index)

    print("Training index")
    target_index.train(target_embeddings)

    print("Adding indices")
    target_index.add(target_embeddings)
    end_time = time.time()
    print(str(end_time - start_time) + " seconds to build index")
    print("Searching")
    start_time = time.time()
    target_index.nprobe = 150
    res_scores, res_indices = batched_search(
        target_index, query_embeddings, k=100
    )

    res_scores = res_scores - score_adjustment
    res_scores[res_scores < 0] = 0
    #res_indices = res_indices[res_scores > 0].astype(np.int64)
    #res_scores = res_scores[res_scores > 0].astype(np.float32)
    end_time = time.time()
    print(str(end_time - start_time) + " seconds to perform FAISS search")
    print("Processing and outing hits")
    start_time = time.time()
    process_hits.process_hits_py(
        np.float32(res_scores),
        np.int64(res_indices),
        query_starts,
        target_starts,
        args.output_path,
        1
    )
    end_time = time.time()
    print(str(end_time - start_time) + " seconds to combine hits")
    print("Done! Total time: " + str(global_start - end_time))

def read_embeddings(
    embedding_path, soft_mask_path
    ):

    embeddings = np.load(embedding_path)
    embeddings = [embeddings[key] for key in sorted(embeddings.keys(), key=int)]
    #print(embeddings[0].shape)
    if soft_mask_path is not None:
        with open(soft_mask_path, "r") as file:
            sequences = list(SeqIO.parse(file, "fasta"))
            # sequences = sorted(sequences, key=lambda x:str(x.id))

            for i, seq in enumerate(sequences):
                mask = np.array(
                    [True if c.isupper() else False for c in str(seq.seq)],
                    dtype=bool,
                )
                embeddings[i] = embeddings[i][mask].copy()
    #print(embeddings[0].shape)
    return embeddings


# --------------------------------------------------
def prepare_embedding_data(embeddings):
    """ Does something """

    seq_ids = []
    lengths = [0]
    #print(embeddings[0].shape)
    for seqid, embedding in enumerate(embeddings):
        #if seqid == 0:
        #    print(embedding.shape)
        embeddings[seqid] = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        seq_ids.extend([seqid for _ in range(embedding.shape[0])])
        lengths.append(embedding.shape[0])

    return np.array(seq_ids), np.concatenate(embeddings, axis=0), np.cumsum(np.array(lengths, dtype=np.int64)[:-1])# - lengths[0]


# --------------------------------------------------
def batched_search(target_index, query_embeddings, k=1000, batch_size=512):
    """ Does something """

    num_queries = query_embeddings.shape[0]
    num_batches = (
        num_queries + batch_size - 1
    ) // batch_size  # Calculate the number of batches

    all_scores = []
    all_indices = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_queries)
        batch_query_embeddings = query_embeddings[start_idx:end_idx]

        res_scores, res_indices = target_index.search(
            batch_query_embeddings, k
        )

        all_scores.append(res_scores)
        all_indices.append(res_indices)

    # Combine results from all batches
    combined_scores = np.vstack(all_scores)
    combined_indices = np.vstack(all_indices)

    return combined_scores, combined_indices


# --------------------------------------------------
if __name__ == "__main__":
    main()
