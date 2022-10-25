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
    model_name = "ResNet1d"
    evaluator_name = "UniRefMeanPoolEvaluator"
    global figure_path
    figure_path = f"contrastive_all_sequences_2_layers_mean_pool.png"

    num_threads = 12
    log_verbosity = logging.INFO
    root = "/xdisk/twheeler/colligan/model_data/contrastive_2_layers/ResNet1d/1/"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    @to_dict
    class evaluator_args:
        model_device = "cuda"
        index_device = "cuda"
        normalize_embeddings = True
        select_random_aminos = False
        figure_path = figure_path
        encoding_func = None
        index_string = "Flat"
        hmmer_hit_file = None  # "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/hits.txt"
        query_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa"
        target_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa"
        # query_file = "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/queries.fa"
        # target_file = "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/targets.fa"
        distance_threshold = 0.5
        evalue_threshold = 10
        minimum_seq_length = 0
        max_seq_length = 100000
