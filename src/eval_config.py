import logging

import yaml
from sacred import Experiment

# from src.datasets.datasets import sanitize_sequence
from src.utils.helpers import to_dict

logger = logging.getLogger("evaluate")
logger.setLevel(logging.WARNING)

evaluation_ex = Experiment()
# convert a class to a dictionary with a decorator


def temporal():
    device = "cuda"
    model_name = "ResNet100M"
    evaluator_name = "TemporalBenchmark"
    print(model_name)

    num_threads = 12
    log_verbosity = logging.DEBUG
    root = "/xdisk/twheeler/colligan/model_data/resnet_meanpool/ResNet1dSequencePool/1"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    @to_dict
    class evaluator_args:
        model_device = "cuda"
        index_device = "cuda"
        n_neighbors = 2048
        embedding_dimension = 256
        num_queries = 6370
        batch_size = 512
        target_db_size = 30e6
        resnet_n_params = "10M"


@evaluation_ex.config
def config():

    device = "cuda"
    model_name = "ResNet1dKmerSampler"
    evaluator_name = "KMerEmbedEvaluator"
    global figure_path

    figure_path = f"length_100_15mer_10k_targets_4_res_blocks.png"

    num_threads = 12
    log_verbosity = logging.INFO
    root = "/xdisk/twheeler/colligan/model_data/ResNet1dKmerSampler/8/"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    @to_dict
    class evaluator_args:
        target_sequence_fasta = None
        blosum = 62
        index_device = "cuda"
        device = "cuda"
        sample_percent = 1.0
        num_queries = 100
        normalize_embeddings = True
        num_targets = 10000
        index_string = "Flat"
        figure_path = figure_path
        query_percent = 1.0
        distance_threshold = 0.0
        kmer_length = 15
        seq_len = 200
