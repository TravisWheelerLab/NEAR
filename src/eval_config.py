import itertools
import logging
import os

import yaml
from sacred import Experiment

from src.data.utils import get_evaluation_data

# from src.datasets.datasets import sanitize_sequence
from src.utils.helpers import to_dict

logger = logging.getLogger("evaluate")
logger.setLevel(logging.WARNING)

evaluation_ex = Experiment()
"""Config file for running evaluation code
Evaluation code is in src/__init__.py"""

HOME = os.environ["HOME"]
# convert a class to a dictionary with a decorator
# ROOT = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation"
ROOT = "/xdisk/twheeler/daphnedemekas/prefilter-output"

if not os.path.exists(ROOT):
    os.mkdir(ROOT)


@evaluation_ex.config
def contrastive_alignments_kmer():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveKmerEvaluator"
    description = "Use CPU with optimization"
    task_id = None
    testing = False
    print(task_id)
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    # val_target_file = open("/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r")
    # val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-kmer"

    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=4,
        save_dir=save_dir,
    )
    print("Loaded all data")
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 10
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )

    if testing:
        querysequences = dict(itertools.islice(querysequences.items(), 0, 100))
        targetsequences = dict(
            itertools.islice(targetsequences.items(), 0, 1000)
        )

    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "Flat",
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 15,
        "max_seq_length": 512,
        "output_path": save_dir,
        "nprobe": 1,
    }


# evaluation_ex.config
def contrastive_blosum_flat():
    device = "cpu"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use CPU with optimization"
    task_id = None
    print(task_id)
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/4"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    val_target_file = open(
        "/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r"
    )
    val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities"

    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=4,
        val_targets=val_targets,
        save_dir=save_dir,
    )
    print("Loaded all data")
    del val_targets
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 100
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )
    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "Flat",
        "figure_path": f"{ROOT}/BlosumEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 15,
        "max_seq_length": 512,
        "output_path": save_dir,
        "nprobe": 1,
    }


# @evaluation_ex.config
def contrastive_blosum_quantized():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use CPU with optimization"
    task_id = None
    query_id = 0
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/4"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    val_target_file = open(
        "/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r"
    )
    val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/BlosumEvaluation/similarities-IVF"

    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=0,
        val_targets=val_targets,
        save_dir=save_dir,
    )
    print("Loaded all data")
    del val_targets
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 100
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )
    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "IVF256,PQ16",
        "figure_path": f"{ROOT}/BlosumEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 0,
        "max_seq_length": 512,
        "output_path": save_dir,
    }


# @evaluation_ex.config
def contrastive_alignments_quantized():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use GPU with optimization query id 0s"
    task_id = None
    query_id = 0
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    val_target_file = open(
        "/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r"
    )
    val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities-IVF-0"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    querysequences, targetsequences, all_hits = get_evaluation_data(
        targethitsfile="/xdisk/twheeler/daphnedemekas/ALL_PHMMERMAX_QUERY0_RESULTS.pkl",
        query_id=query_id,
        save_dir=save_dir,
    )
    print("Loaded all data")
    del val_targets
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 10
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )
    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "IVF256,PQ16",
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 0,
        "max_seq_length": 512,
        "output_path": save_dir,
    }


# @evaluation_ex.config
def contrastive_alignments_flat():
    device = "cpu"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Flat Index query id 0s"
    task_id = None
    query_id = 4
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    val_target_file = open(
        "/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r"
    )
    val_targets = val_target_file.read().splitlines()
    save_dir = "/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/similarities"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    querysequences, targetsequences, all_hits = get_evaluation_data(
        query_id=query_id, save_dir=save_dir
    )
    print("Loaded all data")
    del val_targets
    print(len(querysequences))
    if task_id is not None:
        numqueries = len(querysequences) // 4
        task_id = int(task_id)
        querysequences = dict(
            itertools.islice(
                querysequences.items(),
                int(numqueries * ((task_id) - 1)),
                int(numqueries * ((task_id))),
            )
        )
    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": device,
        "index_string": "Flat",
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 0,
        "max_seq_length": 512,
        "output_path": save_dir,
    }


def temporal():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluatorFile"
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/4"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    @to_dict
    class evaluator_args:
        query_file = f"{HOME}/Q_benchmark2k30k.fa"
        target_file = f"{HOME}/T_benchmark2k30k.fa"
        encoding_func = None
        model_device = "cuda"
        index_device = "cpu"
        figure_path = f"{HOME}/prefilter/ResNet1d/6/test.png"
        normalize_embeddings = True
        minimum_seq_length = 0
        max_seq_length = 1000


# @evaluation_ex.config
def vae():
    device = "cuda"
    model_name = "VAEIndels"
    evaluator_name = "UniRefTiledVAEEvaluator"
    print(model_name)

    num_threads = 12
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/VAEIndels/7"
    checkpoint_path = f"{root}/checkpoints/epoch_99_0.00836.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    @to_dict
    class evaluator_args:
        query_file = f"{HOME}/Q_benchmark2k30k.fa"
        target_file = f"{HOME}/T_benchmark2k30k.fa"
        encoding_func = None
        model_device = "cpu"
        index_device = "cuda"
        figure_path = f"{HOME}/prefilter/VAEIndels/7/fig.png"
        normalize_embeddings = True
        minimum_seq_length = 256
        max_seq_length = 512
        overwrite = True
        n_vae_samples = 1
        tile_size = 256
        tile_step = 128


def config():

    device = "cuda"
    model_name = "ResNet1dKmerSampler"
    evaluator_name = "KMerEmbedEvaluator"
    global figure_path

    figure_path = f"test.png"

    num_threads = 12
    log_verbosity = logging.INFO
    root = "/xdisk/twheeler/colligan/model_data/ResNet1dKmerSampler/6/"
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
        kmer_length = 32
        seq_len = 200


# @evaluation_ex.config
def config():

    device = "cuda"
    model_name = "ResNet2d"
    evaluator_name = "UniRefTiledKmerEvaluator"
    global figure_path
    global encoding_func
    encoding_func = None

    figure_path = f"kmer_contains_untrained_0_to_512.png"

    num_threads = 12
    log_verbosity = logging.INFO
    root = "/xdisk/twheeler/colligan/model_data/kmer_2d_range_of_kmers/ResNet2d/1/"
    checkpoint_path = f"{root}/checkpoints/best_loss_model.ckpt"

    @to_dict
    class evaluator_args:
        model_device = "cuda"
        index_device = "cuda"
        normalize_embeddings = True
        select_random_aminos = False
        figure_path = figure_path
        encoding_func = encoding_func
        index_string = "Flat"
        # "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/hits.txt"
        hmmer_hit_file = None
        query_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa"
        target_file = "/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa"
        # query_file = "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/queries.fa"
        # target_file = "/xdisk/twheeler/colligan/aligned_benchmark/only_alignments/targets.fa"
        distance_threshold = 0.0
        evalue_threshold = 10
        minimum_seq_length = 0
        max_seq_length = 256
        tile_size = 128
        tile_step = 32
        add_random_sequence = False
