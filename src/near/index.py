from .fasta_data import FASTAData
from .models import NEARResNet

import torch
import torch.nn.functional as F
import faiss
import numpy as np

from typing import Literal
from tqdm import tqdm
import time

class NEARIndex:
    def __init__(self):
        self.fasta_data                 = None
        self.index                      = None
        self.index_build_algo           = None
        self.model_dims                 = None
        self.graph_degree               = None
        self.intermediate_graph_degree  = None
        self.nn_descent_niter           = None
        self.itopk_size                 = None
        self.stride                     = None
        self.labels                     = None
        self.real_pos                   = None

    def load_from_path(self, path: str):
        pass

    def save_to_path(self, path:str):
        pass

    def create_index(self,  stride: int = 8,
                            index_build_algo: str = "Default",
                            graph_degree: int = 256,
                            intermediate_graph_degree: int = 512,
                            nn_descent_niter: int = 20,
                            itopk_size:int =64,
                            model_dims: int = 256,
                            verbose: bool = False,
                            device: str = "cuda",
                            residues_per_batch: int = 512*1024,):

        if self.fasta_data is None:
            raise ValueError("FASTA data has not been loaded")

        self.index_build_algo = index_build_algo
        self.model_dims = model_dims
        self.graph_degree = graph_degree
        self.intermediate_graph_degree = intermediate_graph_degree
        self.nn_descent_niter = nn_descent_niter
        self.stride = stride
        self.itopk_size = itopk_size

        # Build the index and initialize the config
        if verbose:
            print("Initializing index...")
        match index_build_algo:
            case "Default" | "GPU_CAGRA_NN_DESCENT":



                config = faiss.GpuIndexCagraConfig()
                config.itopk_size = itopk_size
                config.build_algo = faiss.graph_build_algo_NN_DESCENT
                self.index = faiss.GpuIndexCagra(faiss.StandardGpuResources(), model_dims, faiss.METRIC_INNER_PRODUCT, config)


            case "GPU_CAGRA":
                self.index = faiss.GpuIndexCagra(faiss.StandardGpuResources(), 256, faiss.METRIC_INNER_PRODUCT)

            case _:
                raise ValueError(f"Invalid index type: {index_build_algo}")

        if verbose:
            print(f"Creating embeddings (using device={device})...")
        start_time = time.time()
        embeddings, labels, real_pos = embed_data_with_model(self.fasta_data,
                                                             selection_frequency= self.stride,
                                                             random_selection_rate=1.0,
                                                             device=device,
                                                             verbose=verbose,
                                                             residues_per_batch=residues_per_batch,
                                                             discard_masked=True)
        self.labels = labels
        self.real_pos = real_pos

        if verbose:
            print(f"Time to create embeddings: {time.time() - start_time}")
            print("Adding embeddings to index. This may take a while...")

        start_time = time.time()

        self.index.train(embeddings)
        self.index.add(embeddings)

        if verbose:
            print(f"Time to add embeddings to index: {time.time() - start_time}")

        del embeddings





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
        Path where the index/metadata should be saved

    Notes
    ----------

    """




IndexType = Literal["Default", "GPU_CAGRA", "GPU_CAGRA_NN_DESCENT"]

def embed_data_with_model(model: NEARResNet,
                          data: FASTAData,
                          selection_frequency: int = 16,
                          random_selection_rate: float = 1.0,
                          discard_masked : bool = True,
                          device: str = "cuda",
                          residues_per_batch: int = 512*1024,
                          verbose=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Embeds `data` with `model` with some (optional) level of sparsity.
    By default, masked embeddings will be discarded.

    Parameters
    ----------
    model : NEARResNet
        The model used for embedding
    data : FASTAData
        The data that will be embedded
    selection_frequency : int
        How many embeddings are selected for each embedding discarded.
        A value of 0 means all embeddings will be kept
        A value of 1 means that every other embedding will be kept
        A value of 16 means that every 16th embedding will be kept
    random_selection_rate : float
        The percentage of embeddings that will be (randomly) kept
        A value of 1 means that there will be no random selection
        A value of 0.5 means that only half of embeddings will be selected
    discard_masked : bool
        If True, masked embeddings are discarded
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
        if selection_frequency > 0:
            half_freq = (selection_frequency + 1) // 2

            num_emb = (length - selection_frequency) // selection_frequency
            included_pos = torch.linspace(half_freq, length - half_freq, num_emb, dtype=torch.long)
            discard_mask = torch.zeros(mask.shape[1], dtype=bool)
            discard_mask[included_pos] = True

            mask = torch.logical_and(mask, discard_mask.unsqueeze(0))

        if random_selection_rate < 1.0:
            random_selection_mask = torch.rand(*mask.shape) < random_selection_rate
            mask = torch.logical_and(mask, random_selection_mask)

        masks_by_length[length] = mask
        total_embeddings += mask.sum()

    if verbose:
        print(f"{total_embeddings} embeddings will be stored")
        print(f"Allocating embedding memory on '{device}'...")

    embeddings = torch.zeros(total_embeddings, 256, device=device)
    labels = np.zeros(total_embeddings, dtype=np.uint64)
    real_pos = np.zeros(total_embeddings, dtype=np.int32)

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
            labels[num_embeddings:num_embeddings+len(length_labels)] = length_labels

            # Calculate embeddings in batches
            batch_size = (residues_per_batch // length) + 2
            for i in range(0, token_tensors.shape[0], batch_size):
                #Gather the tokens/masks
                end = min(i+batch_size, token_tensors.shape[0])
                batch_tokens = token_tensors[i:end]

                batch_mask = mask[i:end]
                rp = (batch_mask.cumsum(-1)[batch_mask] - 1).int().cpu().numpy()
                batch_mask = batch_mask.flatten()
                #Embed and transpose, then mask and normalize
                batch_embeddings = model(batch_tokens).transpose(-1, -2).flatten(start_dim=0, end_dim=-2)[batch_mask]

                batch_embeddings = F.normalize(batch_embeddings, dim=-1)

                # Assign embeddings and continue
                embeddings[num_embeddings:num_embeddings+len(batch_embeddings)] = batch_embeddings
                real_pos[num_embeddings:num_embeddings + len(batch_embeddings)] = rp
                num_embeddings += len(batch_embeddings)
    if verbose:
        print("Done.")

    return embeddings, labels, real_pos