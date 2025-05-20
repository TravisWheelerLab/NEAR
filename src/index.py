from src.fasta_data import FASTAData
from src.models import NEARResNet

import torch
import torch.nn.functional as F
import faiss
import numpy as np

from typing import Literal
from tqdm import tqdm

def load_index(file_path: str) -> faiss.Index:
    """
    Load a search index from file

    Parameters
    ----------
    file_path : str
        The saved index file

    Notes
    ----------
    For now this just calls `faiss.read_index`, but it may perform additional
    actions in the future. This function should be used instead of manually loading
    the faiss index.
    """

    return faiss.read_index(file_path)


def save_index(index: faiss.Index, file_path: str):
    """
    Saves a FAISS index to a file

    Parameters
    ----------
    file_path : str
        Path where the index should be saved

    Notes
    ----------
    For now this just calls `faiss.write_index`, but it may perform additional
    actions in the future. This function should be used instead of manually saving
    the faiss index.
    """

    faiss.write_index(index, file_path)


IndexType = Literal["Default", "GPU_CAGRA", "GPU_CAGRA_NN_DESCENT"]


def create_index(index_type: IndexType = "Default",
                 graph_degree: int = 64,
                 nn_descent_niter: int = 20) -> faiss.Index:
    """
    Creates a search index without filling it.

    Parameters
    ----------
    index_type : IndexType
        The type of index to create. Must be one of:
        - "Default": Currently defaults to GPU_CAGRA_NN_DESCENT
        - "GPU_CAGRA": GPU-based Cagra index with default configuration
        - "GPU_CAGRA_NN_DESCENT": GPU-based Cagra index trained with NN descent
    graph_degree : int
        The degree of the graph for the final Cagra index
        Increasing the degree will increase accuracy, but also increase size/build-time
        Note that the top-k search parameter cannot be larger than the graph degree
    nn_descent_niter : int
        The number of NN descent iterations to use when building the graph with GPU_CAGRA_NN_DESCENT
    """

    match IndexType:
        case "Default" | "GPU_CAGRA_NN_DESCENT":
            config = faiss.GpuIndexCagraConfig()
            config.build_algo = faiss.graph_build_algo_NN_DESCENT
            index = faiss.GpuIndexCagra(faiss.StandardGpuResources(), 256, faiss.METRIC_INNER_PRODUCT, config)

        case "GPU_CAGRA":
            index = faiss.GpuIndexCagra(faiss.StandardGpuResources(), 256, faiss.METRIC_INNER_PRODUCT)

        case _:
            raise ValueError(f"Invalid index type: {index_type}")

    return index

def fill_index(index: faiss.Index, embeddings: torch.tensor):
    index.train(embeddings.float())
    index.add(embeddings.float())

def embed_data_with_model(model: NEARResNet,
                          data: FASTAData,
                          discard_frequency: int = 16,
                          discard_masked : bool = False,
                          device: str = "cuda",
                          residues_per_batch: int =512*1024,
                          verbose=True) -> tuple[torch.Tensor[float], torch.Tensor[torch.uint64]]:
    """
    Embeds `data` with `model` with some (optional) level of sparsity.
    By default, masked embeddings will be discarded.

    Parameters
    ----------
    model : NEARResNet
        The model used for embedding
    data : FASTAData
        The data that will be embedded
    discard_frequency : int
        How many embeddings are discarded for each embedding kept.
        A value of 0 means all embeddings will be kept
        A value of 1 means that every other embedding will be kept
        A value of 16 means that every 16th embedding will be kept
    discard_masked : bool
        If True, masked embeddings will be discarded
    device : str
        The device being used for embedding data
    residues_per_batch : int
        The (approximate) number of residues in a single batch
    verbose : bool
        If true progress will be output and a progress bar will be displayed

    Returns
    ----------
    Returns a tuple of embeddings and corresponding embedding IDs.
    """

    masks_by_length = {}
    seq_lengths = data.tokens_by_length.keys()

    if verbose:
        print("Calculating masks...")
        seq_lengths = tqdm(seq_lengths)

    total_embeddings = 0
    for length in seq_lengths:
        # Calculate the mask that we will be using
        # Check to see if we are using the lower-case/X masking
        if not discard_masked:
            mask = torch.ones_like(data.tokens_by_length[length])
        else:
            mask = data.masks_by_length[length].clone()

        # Apply the discard to the masks
        if discard_frequency > 0:
            half_freq = (discard_frequency + 1) // 2
            discard_mask = torch.zeros(mask.shape[1], dtype=bool)

            discard_mask[half_freq:-half_freq:discard_frequency+1] = True
            discard_mask[-half_freq] = True

            mask = torch.logical_and(mask, discard_mask.unsqueeze(0))

        masks_by_length[length] = mask
        total_embeddings += mask.sum()

    if verbose:
        print(f"{total_embeddings} embeddings will be stored")
        print(f"Allocating embedding memory on '{device}'...")

    embeddings = torch.zeros(total_embeddings, 256, device=device)
    labels = np.zeros(total_embeddings, dtype=torch.uint64)

    if verbose:
        print("Pushing model to device...")

    model.to(device)
    model.half()
    model.eval()

    if verbose:
        print("Creating embeddings...")
        seq_lengths = tqdm(data.tokens_by_length.keys())

    with torch.no_grad():
        num_embeddings = 0
        for length in seq_lengths:
            #Gather the data for this length
            token_tensors = data.tokens_by_length[length].to(device)
            mask = masks_by_length[length]
            length_labels = data.tokenids_by_length[length][mask].flatten()
            labels[num_embeddings:num_embeddings+length_labels.shape[0]] = length_labels

            # Calculate embeddings in batches
            batch_size = (residues_per_batch // length) + 2
            for i in range(0, token_tensors.shape[1], batch_size):
                #Gather the tokens/masks
                batch_tokens = token_tensors[i:i+batch_size]
                batch_mask = mask[i:i+batch_size].flatten()

                #Embed and transpose, then mask and normalize
                batch_embeddings = model(batch_tokens).transpose(-1, -2).flatten(start_dim=1)
                batch_embeddings = F.normalize(batch_embeddings[batch_mask], dim=-1)

                # Assign embeddings and continue
                embeddings[num_embeddings:num_embeddings+batch_embeddings.shape[0]] = batch_embeddings.flatten(start_dim=1)
                num_embeddings += batch_embeddings.shape[0]
    if verbose:
        print("Done.")

    return embeddings, labels