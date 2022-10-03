import logging
import pdb
from copy import deepcopy
from glob import glob

import faiss
import torch
import yaml
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
    model_name = "SequenceVAEWithIndels"
    evaluator_name = "UniRefRandomUngappedVAEEvaluator"

    num_threads = 12
    log_verbosity = logging.INFO
    root = "/xdisk/twheeler/colligan/indel_model_data/train_cnn/SequenceVAETrainCNN/4/"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    print(hparams)

    @to_dict
    class evaluator_args:
        overwrite = True
        model_device = "cuda"
        index_device = "cuda"
        figure_path = f"tile_step75train_cnn.png"
        normalize_embeddings = True
        select_random_aminos = False
        encoding_func = None
        index_string = "Flat"
        hit_filename = "none.txt"
        tile_size = 32
        tile_step = 16
        # query_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa"
        # target_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa"
        query_file = (
            "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/queries.fa"
        )
        target_file = (
            "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/targets.fa"
        )
        hit_file = "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/hits.txt"
        n_vae_samples = 1
        distance_threshold = 0.2
        evalue_threshold = 10
        max_seq_length = 512
        random_length = 200
        minimum_seq_length = 256
        n_neighbors = 100

    # @to_dict
    # class evaluator_args:
    #     overwrite = True
    #     encoding_func = None
    #     select_random_aminos = False
    #     normalize_embeddings = True
    #     hit_filename = "test.txt"
    #     model_device = "cuda"
    #     index_device = "cuda"
    #     figure_path = f"raw_sequences_tilestep1.png"
    #     index_string = "Flat"
    #     # query_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa"
    #     # target_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa"
    #     query_file = "/xdisk/twheeler/colligan/aligned_benchmark/raw_sequences/queries.fa"
    #     target_file = "/xdisk/twheeler/colligan/aligned_benchmark/raw_sequences/targets.fa"
    #     n_vae_samples = 1
    #     minimum_seq_length = 256
    #     tile_size = 256
    #     tile_step = 1
    #     evalue_threshold = 10
    #     distance_threshold = 0.3
    #     max_seq_length = 10000
    #     n_neighbors = 100

    # @to_dict
    # this is for SyntheticEvaluator
    # class evaluator_args:
    #     target_sequence_fasta = "/xdisk/twheeler/colligan/esl_random/targets.fa"
    #     num_queries = 20
    #     blosum = 45
    #     figure_path = "test.png"
    #     device = "cuda"
    #     sample_percent = 1.0
    #     query_percent = 1.0
    #     normalize_embeddings = True
    #     index_string = "Flat"
    #     index_device = "cuda"
    #     distance_threshold = 0.2
    #     # distance_threshold = 0.1
