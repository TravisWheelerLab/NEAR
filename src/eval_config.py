import logging
import pdb
from copy import deepcopy
from glob import glob

import faiss
import torch
from sacred import Experiment

# from src.datasets.datasets import sanitize_sequence
from src.utils.helpers import non_default_collate, to_dict

logger = logging.getLogger("evaluate")
logger.setLevel(logging.WARNING)

evaluation_ex = Experiment()

# convert a class to a dictionary with a decorator


@evaluation_ex.config
def config():

    device = "cuda"
    model_name = "SequenceVAE"
    evaluator_name = "SyntheticVAEEvaluator"

    @to_dict
    class evaluator_args:
        target_sequence_fasta = f"/xdisk/twheeler/colligan/targets.fa"
        # target_sequence_fasta = f"/Users/mac/share/prefilter/test.fa"
        blosum = 90
        num_queries = 20
        device = "cuda"
        sample_percent = 1.0
        normalize_embeddings = True
        index_string = "Flat"
        index_device = "cuda"
        query_percent = 1.0
        distance_threshold = 0.4

    num_threads = 12

    log_verbosity = logging.WARNING
    checkpoint_path = f"/xdisk/twheeler/colligan/model_data/SequenceVAE/5/checkpoints/best_loss_model.ckpt"
