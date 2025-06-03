from .fasta_data import FASTAData
from .models import NEARResNet
from .search_processor import AsyncNearResultsProcessor

import tqdm
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import sys
from time import sleep

def search_against_index(output_file_path: str,
                         model: NEARResNet,
                         index: faiss.Index,
                         query_data: FASTAData,
                         target_data: FASTAData,
                         target_labels: np.array[np.uint64],
                         filter1,
                         filter2,
                         sparsity,
                         angle_deviation_data,
                         stats,
                         device: str = "cuda",
                         residues_per_batch: int = 512*1024,
                         verbose=False,
                         top_k=64):
    """
    Searches for matches in the FAISS index using the given model, data, and index.

    Parameters
    ----------
    output_file_path : str
        Path where search results will be saved
    model : NEARResNet
        The model used for embedding query sequences
    query_data : FASTAData
        The query sequences to search with
    target_data : FASTAData
        The query sequences to search with
    index : faiss.Index
        The FAISS index to search against
    target_labels : np.array[np.uint64]
        The labels corresponding to entries in the index
    device : str
        The device being used for embedding data
    residues_per_batch : int
        The (approximate) number of residues in a single batch
    top_k : int
        Number of nearest neighbors to return for each query
        Note that the top-k search parameter cannot be larger than the graph degree
    verbose : bool
        If true progress will be output and a progress bar will be displayed
    """
    # Prepare the model / move the model
    if verbose:
        print(f"Moving model to {device}...")

    model.to(device)
    model.half()
    model.eval()

    if verbose:
        print("Calculating target lengths...")

    target_lengths = np.bincount((target_labels >> 32).astype(np.int64))

    if verbose:
        print("Calculating query lengths...")

    query_lengths = np.zeros(len(query_data.seqid_to_name), dtype=np.uint64)
    for length in query_data.masks_by_length.keys():
        query_masks = query_data.masks_by_length[length].sum(-1).numpy()
        query_indices = query_data.tokenids_by_length[length][:,0] >> 32
        query_lengths[query_indices] += query_masks

    lengths = query_data.tokens_by_length.keys()
    if verbose:
        print(f"Searching sequences...")
        lengths = tqdm(lengths)

    # Create the AsyncNearResultsProcessor
    if verbose:
        print("Creating NEAR search results processor...")
    try:
        near_processor = AsyncNearResultsProcessor(output_file_path,
                                                   query_data,
                                                   target_data,
                                                   query_lengths,
                                                   target_lengths,
                                                   top_k,
                                                   filter1,
                                                   filter2,
                                                   sparsity,
                                                   angle_deviation_data,
                                                   stats,)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    try:
        for length in lengths:
            # Gather the data that we will be processing in this length bin
            token_tensors = query_data.tokens_by_length[length].to(device)
            mask = query_data.masks_by_length[length].clone()
            token_ids = query_data.tokenids_by_length[length]

            batch_size = (residues_per_batch // length) + 2

            with torch.no_grad():
                for i in range(0, token_tensors.shape[1], batch_size):
                    batch_tokens = token_tensors[i:i + batch_size]
                    batch_mask = mask[i:i + batch_size].flatten()
                    embeddings = model(batch_tokens).transpose(-1, -2).flatten(start_dim=0, end_dim=-2)
                    embeddings = F.normalize(embeddings[batch_mask], dim=-1)
                    query_ids = token_ids[i:i+batch_size].flatten()[batch_mask]

                    scores, indices = index.search(embeddings, k=top_k)
                    indices = target_labels[indices]
                    near_processor.add_to_queue(query_ids, indices, scores)
        if verbose:
            print("Search is complete.")
        while near_processor.not_done():
            if verbose:
                print("Waiting for search processor...")
            sleep(0.1)

    finally:
        near_processor.finalize()