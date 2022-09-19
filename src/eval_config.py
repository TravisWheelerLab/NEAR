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

    device = "cpu"
    index_device = "cuda"
    overwrite = False
    n_neighbors = 100
    # distance threshold to use for faiss and
    # filtering
    distance_threshold = 0.7
    normalize_embeddings = True
    sample_percent = 1.0
    query_percent = 1
    nprobe = 1
    istr = "Flat"
    select_random_aminos = False
    hit_filename = f"vae_15_hits.txt"
    num_threads = 32
    model_name = "SequenceVAE"
    evaluator_name = "UniRefVAEEvaluator"
    root = "/xdisk/twheeler/colligan/"
    seq_len = 32

    query_file = (
        f"{root}/data/prefilter/uniref_benchmark/{model_name}/Q_benchmark2k30k.fa"
    )
    target_file = (
        f"{root}/data/prefilter/uniref_benchmark/{model_name}/T_benchmark2k30k.fa"
    )

    log_verbosity = logging.INFO
    checkpoint_path = f"/xdisk/twheeler/colligan/model_data/SequenceVAE/13/checkpoints/best_loss_model.ckpt"

    cnn_model_state_dict = f"{root}/prefilter/model_16.sdic"

    model_args = {
        "learning_rate": 1e-4,
        "log_interval": 100,
        "cnn_model_state_dict": cnn_model_state_dict,
        "initial_seq_len": seq_len,
    }

    evaluator_args = {
        "query_file": query_file,
        "overwrite": overwrite,
        "seq_len": seq_len,
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
