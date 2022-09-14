import logging
import pdb
from copy import deepcopy
from glob import glob

import faiss
import torch
from sacred import Experiment

# from src.datasets.datasets import sanitize_sequence
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

    device = "cuda"
    index_device = "cuda"
    n_neighbors = 100
    # distance threshold to use for faiss and
    # filtering
    distance_threshold = 0.68
    normalize_embeddings = False
    sample_percent = 0.2
    nprobe = 1
    istr = "Flat"
    select_random_aminos = False
    query_percent = 1.0
    hit_filename = f"query_100pct_10pcttarget_db.txt"
    num_threads = 32
    model_name = "ResNet"
    evaluator_name = "UniRefFaissEvaluator"
    root = "/xdisk/twheeler/colligan/"

    query_file = (
        f"{root}/data/prefilter/uniref_benchmark/{model_name}/Q_benchmark2k30k.fa"
    )
    target_file = (
        f"{root}/data/prefilter/uniref_benchmark/{model_name}/T_benchmark2k30k.fa"
    )

    log_verbosity = logging.INFO

    checkpoint_path = f"{root}/data/prefilter/model_16.sdic"
    # checkpoint_path = f"model_data/sept6/SequenceVAE/1/checkpoints/best_loss_model.ckpt"

    cnn_model_state_dict = f"{root}/prefilter/model_16.sdic"
    model_args = {
        "emb_dim": 256,
        "blocks": 5,
        "block_layers": 2,
        "first_kernel": 11,
        "kernel_size": 5,
        "groups": 2,
        "padding_mode": "reflect",
    }

    # model_args = {
    #     "learning_rate": 1e-4,
    #     "log_interval": 100,
    #     "cnn_model_state_dict": cnn_model_state_dict,
    #     "cnn_model_args": cnn_model_args,
    # }

    evaluator_args = {
        "query_file": query_file,
        "nprobe": nprobe,
        "target_file": target_file,
        "normalize_embeddings": normalize_embeddings,
        "encoding_func": None,
        "index_device": index_device,
        "index_string": istr,
        "n_neighbors": n_neighbors,
        "distance_threshold": distance_threshold,
        "hit_filename": hit_filename,
        "sample_percent": sample_percent,
        "model_device": device,
        "select_random_aminos": select_random_aminos,
        "query_percent": query_percent,
    }
