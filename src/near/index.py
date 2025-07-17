from .fasta_data import FASTAData
from .models import NEARResNet
from .search import search_against_index

import torch
import torch.nn.functional as F
import faiss
import numpy as np

from typing import Literal
from tqdm import tqdm
import time

import os
import pickle
from typing import Optional, Tuple, Dict, Any, Union


class NEARIndex:
    def __init__(self, fasta_data = None):
        self.fasta_data                 = fasta_data
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

    def save_to_path(self, path: str) -> bool:
        """
        Save the NEARIndex instance to a single file, including the FAISS index and all metadata.

        Parameters
        ----------
        path : str
            Path where the index and metadata should be saved

        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        try:
            # Verify that all necessary components are present
            if self.index is None:
                print("Error: No index to save")
                return False

            if self.labels is None:
                print("Error: No labels to save")
                return False

            if self.real_pos is None:
                print("Error: No real_pos data to save")
                return False

            # Required configuration parameters
            required_params = [
                'index_build_algo', 'model_dims', 'graph_degree',
                'intermediate_graph_degree', 'nn_descent_niter',
                'itopk_size', 'stride'
            ]

            for param in required_params:
                if getattr(self, param) is None:
                    print(f"Error: Missing required parameter '{param}'")
                    return False

            # Convert FAISS index to bytes
            index_bytes = faiss.serialize_index(self.index)

            # Prepare data dictionary
            save_data = {
                'index_bytes': index_bytes,
                'index_build_algo': self.index_build_algo,
                'model_dims': self.model_dims,
                'graph_degree': self.graph_degree,
                'intermediate_graph_degree': self.intermediate_graph_degree,
                'nn_descent_niter': self.nn_descent_niter,
                'itopk_size': self.itopk_size,
                'stride': self.stride,
                'labels': self.labels,
                'real_pos': self.real_pos,
                'format_version': 1  # For future compatibility checks
            }

            # Save the fasta_data reference separately if it exists
            # We don't include it in the main dictionary to avoid large serializations
            if self.fasta_data is not None:
                save_data['has_fasta_data'] = True
            else:
                save_data['has_fasta_data'] = False

            # Save everything to a single file
            with open(path, 'wb') as f:
                pickle.dump(save_data, f)

            # If we have FASTA data and it has a save method, save it separately
            if self.fasta_data is not None and hasattr(self.fasta_data, 'save'):
                fasta_path = f"{path}.fasta_data"
                self.fasta_data.save(fasta_path)

            return True

        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False

    def load_from_path(self, path: str) -> bool:
        """
        Load a NEARIndex instance from a file, with type checking and validation.

        Parameters
        ----------
        path : str
            Path where the index and metadata are saved

        Returns
        -------
        bool
            True if load was successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                print(f"Error: File not found: {path}")
                return False

            # Load the data dictionary
            with open(path, 'rb') as f:
                save_data = pickle.load(f)

            # Validate format version for compatibility
            if 'format_version' not in save_data or save_data['format_version'] != 1:
                print("Error: Incompatible save format or corrupted file")
                return False

            # Required keys
            required_keys = [
                'index_bytes', 'index_build_algo', 'model_dims', 'graph_degree',
                'intermediate_graph_degree', 'nn_descent_niter', 'itopk_size',
                'stride', 'labels', 'real_pos'
            ]

            # Verify all required keys exist
            for key in required_keys:
                if key not in save_data:
                    print(f"Error: Missing required data '{key}'")
                    return False

            # Type checking
            type_checks = {
                'index_build_algo': (str,),
                'model_dims': (int,),
                'graph_degree': (int,),
                'intermediate_graph_degree': (int,),
                'nn_descent_niter': (int,),
                'itopk_size': (int,),
                'stride': (int,),
                'labels': (np.ndarray,),
                'real_pos': (np.ndarray,),
                'index_bytes': (bytes,),
            }

            for key, expected_types in type_checks.items():
                if not isinstance(save_data[key], expected_types):
                    print(f"Error: Invalid type for '{key}'. Expected {expected_types}, got {type(save_data[key])}")
                    return False

            # Deserialize the FAISS index
            try:
                self.index = faiss.deserialize_index(save_data['index_bytes'])
            except Exception as e:
                print(f"Error deserializing FAISS index: {str(e)}")
                return False

            # Load other attributes
            self.index_build_algo = save_data['index_build_algo']
            self.model_dims = save_data['model_dims']
            self.graph_degree = save_data['graph_degree']
            self.intermediate_graph_degree = save_data['intermediate_graph_degree']
            self.nn_descent_niter = save_data['nn_descent_niter']
            self.itopk_size = save_data['itopk_size']
            self.stride = save_data['stride']
            self.labels = save_data['labels']
            self.real_pos = save_data['real_pos']

            # Load FASTA data if it exists
            if save_data.get('has_fasta_data', False):
                fasta_path = f"{path}.fasta_data"
                if os.path.exists(fasta_path) and hasattr(self.fasta_data.__class__, 'load'):
                    try:
                        self.fasta_data = self.fasta_data.__class__.load(fasta_path)
                    except Exception as e:
                        print(f"Warning: Could not load FASTA data: {str(e)}")
                        self.fasta_data = None
                else:
                    print("Warning: FASTA data was marked as saved but could not be loaded")
                    self.fasta_data = None

            return True

        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False

    def create_index(self,  model,
                            stride: int = 8,
                            index_build_algo: str = "Default",
                            graph_degree: int = 256,
                            intermediate_graph_degree: int = 512,
                            nn_descent_niter: int = 100,
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
        embeddings, labels, real_pos = embed_data_with_model(model,
                                                             self.fasta_data,
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

        embeddings = embeddings.to('cpu').to(float).numpy()
        self.index.train(embeddings)
        self.index.add(embeddings)

        if verbose:
            print(f"Time to add embeddings to index: {time.time() - start_time}")

        del embeddings

    def search_with_query(self,  query_data: FASTAData,
                                 model: NEARResNet,
                                 output_file_path: str,
                                 angle_deviation_data: np.ndarray,
                                 distribution_params: np.ndarray,
                                 filter1: float,
                                 filter2: float,
                                 top_k: int = 64,
                                 query_sparsity: float = 0.0,
                                 residues_per_batch: int = 512 * 1024,
                                 device: str = "cuda",
                                 verbose: bool = False) -> bool:
        """
        Performs a search using the provided query data against the built index.
        
        Parameters
        ----------
        query_data : FASTAData
            The query sequences to search with
        model : NEARResNet
            The model used for embedding query sequences
        output_file_path : str
            Path where search results will be saved
        angle_deviation_data : np.ndarray
            Data related to angle deviations for filtering
        distribution_params : np.ndarray
            Statistical parameters for result filtering
        filter1 : float
            First filter threshold value
        filter2 : float
            Second filter threshold value
        top_k : int
            Number of nearest neighbors to return for each query
        query_sparsity : float
            Sparsity level for query embeddings
        residues_per_batch : int
            The (approximate) number of residues in a single batch
        device : str
            The device being used for embedding data
        verbose : bool
            If true progress will be output and a progress bar will be displayed
        
        Returns
        -------
        bool
            True if search was successful, False otherwise
        """
        # Parameter validation
        if self.index is None:
            if verbose:
                print("Error: No index has been created or loaded")
            return False
        
        if query_data is None:
            if verbose:
                print("Error: No query data provided")
            return False
        
        if model is None:
            if verbose:
                print("Error: No model provided")
            return False
        
        if output_file_path is None or not isinstance(output_file_path, str):
            if verbose:
                print("Error: Invalid output file path")
            return False
        
        # Check if top_k exceeds the graph degree (important constraint for FAISS)
        if top_k > self.graph_degree:
            if verbose:
                print(f"Warning: top_k ({top_k}) cannot be larger than graph degree ({self.graph_degree})")
                print(f"Reducing top_k to {self.graph_degree}")
            top_k = self.graph_degree
        
        # Provide strong warning if both index and query use sparsity
        if query_sparsity > 0 and self.stride > 0:
            print("⚠️ WARNING: BOTH QUERY AND INDEX HAVE SPARSITY ENABLED ⚠️")
            print("This will significantly reduce search quality and recall as embeddings from")
            print("both query and index are being subsampled. Consider disabling one of them.")
            print(f"Current settings: query_sparsity={query_sparsity}, index_stride={self.stride}")
            print("==============================================================================")
        
        try:
            stride = self.stride
            if query_sparsity > 0:
                stride = query_sparsity
            # Perform the search by calling search_against_index
            search_against_index(output_file_path=output_file_path,
                                 model=model,
                                 index=self.index,
                                 query_data=query_data,
                                 target_data=self.fasta_data,
                                 target_labels=self.labels,
                                 target_realpos=self.real_pos,
                                 filter1=filter1,
                                 filter2=filter2,
                                 sparsity=stride,
                                 angle_deviation_data=angle_deviation_data,
                                 stats=distribution_params,
                                 query_sparsity=(query_sparsity > 0),
                                 device=device,
                                 residues_per_batch=residues_per_batch,
                                 verbose=verbose,
                                 top_k=top_k)
            return True
        except Exception as e:
            if verbose:
                print(f"Error during search: {str(e)}")
            return False

def embed_data_with_model(model: NEARResNet,
                          data: FASTAData,
                          selection_frequency: int = 16,
                          random_selection_rate: float = 1.0,
                          discard_masked : bool = True,
                          device: str = "cuda",
                          residues_per_batch: int = 512*1024,
                          verbose=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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