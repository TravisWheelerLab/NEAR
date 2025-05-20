from src.near.fasta_data import FASTAData
from src.near.models import NEARResNet

import faiss
import numpy as np


def search_against_index(output_file_path: str,
                         model: NEARResNet,
                         data: FASTAData,
                         index: faiss.Index,
                         target_labels: np.array[np.uint64],
                         device: str = "cuda",
                         residues_per_batch: int =512*1024,
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
    data : FASTAData
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

