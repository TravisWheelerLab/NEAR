from src.fasta_data import FASTAData
from src.models import NEARResNet

import torch
import faiss

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


def create_index(index_type: IndexType = "Default") -> faiss.Index:
    """
    Creates a search index without filling it.

    Parameters
    ----------
    index_type : IndexType
        The type of index to create. Must be one of:
        - "Default": Currently defaults to GPU_CAGRA_NN_DESCENT
        - "GPU_CAGRA": GPU-based Cagra index with default configuration
        - "GPU_CAGRA_NN_DESCENT": GPU-based Cagra index trained with NN descent
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

def embed_data_with_model(model: NEARResNet,
                          data: FASTAData,
                          discard_frequency: int = 16,
                          discard_masked : bool = True,
                          device: str = "cuda",
                          progress_bar=True) -> tuple[torch.Tensor[float], torch.Tensor[torch.uint64]]:
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
    progress_bar : bool
        If true, then a progress bar will be displayed

    Returns
    ----------
    Returns a tuple of embeddings and corresponding embedding IDs.
    """

    masks_by_length = {}
    seq_lengths = data.tokens_by_length.keys()

    if progress_bar:
        print("Calculating masks...")
        seq_lengths = tqdm(seq_lengths)

    total_embeddings = 0
    for length in seq_lengths:
        if not discard_masked:
            mask = torch.ones_like(data.tokens_by_length[length])
        else:
            mask = data.masks_by_length[length].clone()
        if discard_frequency > 0:
            half_freq = (discard_frequency + 1) // 2
            discard_mask = torch.zeros(mask.shape[1])

            discard_mask[half_freq:-half_freq:discard_frequency+1] = True
            discard_mask[-half_freq] = True

            mask = torch.logical_and(mask, discard_mask.unsqueeze(0))

        total_embeddings += mask.sum()

    if progress_bar:
        print(f"{total_embeddings} embeddings will be stored")
        print(f"Allocating embedding memory on '{device}'...")

    embeddings = torch.zeros(total_embeddings, 256, device=device)
    labels = torch.zeros(total_embeddings, dtype=torch.uint64, device=device)

    if progress_bar:
        print("Creating embeddings...")
        seq_lengths = tqdm(data.tokens_by_length.keys())

    num_embeddings = 0
    for length in seq_lengths:

