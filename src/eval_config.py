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
    evaluator_name = "UniRefTiledVAEEvaluator"

    @to_dict
    class evaluator_args:
        overwrite = True
        seq_len = 256
        hit_filename = "test.txt"
        model_device = "cuda"
        index_device = "cuda"
        tile_size = 256
        tile_step = 100
        evalue_threshold = 10
        figure_path = f"only_contrastive_tilestep100_vae1_evalue10_larger_sequences.png"
        normalize_embeddings = True
        index_string = "Flat"
        distance_threshold = 0.3
        max_seq_length = 512
        query_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa"
        target_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa"
        # query_file = "/xdisk/twheeler/colligan/queries.fa"
        # target_file = "/xdisk/twheeler/colligan/targets.fa"
        # hit_file = "/xdisk/twheeler/colligan/hits.txt"
        n_vae_samples = 1
        select_random_aminos = False
        minimum_seq_length = 256
        encoding_func = None
        n_neighbors = 100

    # @to_dict
    # this is for SyntheticEvaluator
    # class evaluator_args:
    #     target_sequence_fasta = "/xdisk/twheeler/colligan/esl_random/targets.fa"
    #     num_queries = 20
    #     blosum = 45
    #     # figure_path = "test.png"
    #     device = "cuda"
    #     sample_percent = 1.0
    #     query_percent = 1.0
    #     normalize_embeddings = True
    #     index_string = "Flat"
    #     index_device = "cuda"
    #     distance_threshold = 0.0
    #     # distance_threshold = 0.1

    num_threads = 12

    log_verbosity = logging.INFO
    checkpoint_path = f"model_data/SequenceVAE/3/checkpoints/best_loss_model.ckpt"
