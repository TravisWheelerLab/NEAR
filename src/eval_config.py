import logging
import pdb
from copy import deepcopy
from glob import glob

import faiss
import torch
from sacred import Experiment

from src.datasets.datasets import sanitize_sequence
from src.utils import char_to_index
from src.utils.helpers import non_default_collate

logger = logging.getLogger("evaluate")
logger.setLevel(logging.WARNING)

evaluation_ex = Experiment()


def wraps(device):
    def encode(sequence):
        seq = sanitize_sequence(sequence)
        seq = torch.as_tensor([char_to_index[c] for c in seq]).to(device)
        return seq

    return encode


@evaluation_ex.config
def config():

    device = "cpu"
    index_device = "cpu"
    n_neighbors = 10
    distance_threshold = 0.5
    normalize_embeddings = False
    use_faiss = True
    istr = "HNSW32"
    hit_filename = f"with_faiss_{istr}.txt"
    num_threads = 32
    query_file = "/home/u4/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa"
    target_file = (
        "/home/u4/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa"
    )
    log_verbosity = logging.DEBUG

    model_name = "ResNet"
    evaluator_name = "UniRefEvaluator"

    checkpoint_path = "/home/u4/colligan/data/prefilter/model_16.sdic"

    model_args = {
        "emb_dim": 256,
        "blocks": 5,
        "block_layers": 2,
        "first_kernel": 11,
        "kernel_size": 5,
        "groups": 2,
        "padding_mode": "reflect",
    }

    evaluator_args = {
        "query_file": query_file,
        "target_file": target_file,
        "normalize_embeddings": normalize_embeddings,
        "encoding_func": wraps(device),
        "use_faiss": use_faiss,
        "index_device": index_device,
        "index_string": istr,
        "n_neighbors": n_neighbors,
        "distance_threshold": distance_threshold,
        "hit_filename": hit_filename,
        "model_device": device,
    }
