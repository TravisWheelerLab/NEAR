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


# convert a class to a dictionary with a decorator
def to_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


@evaluation_ex.config
def config():

    device = "cuda"
    model_name = "SequenceVAE"
    evaluator_name = "SyntheticVAEEvaluator"

    @to_dict
    class evaluator_args:
        target_sequence_fasta = f"/xdisk/twheeler/colligan/targets.fa"
        blosum = 62
        num_queries = 20
        sequence_length = 32
        device = "cuda"
        sample_percent = 1.0
        normalize_embeddings = True
        index_string = "Flat"
        index_device = "cuda"
        query_percent = 1.0
        distance_threshold = 0.5

    num_threads = 12

    log_verbosity = logging.INFO
    checkpoint_path = f"/xdisk/twheeler/colligan/model_data/SequenceVAE/1/checkpoints/best_loss_model.ckpt"
