import itertools
import logging
import os

import yaml
from sacred import Experiment

from src.data.utils import get_evaluation_data_old as get_evaluation_data

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

#@evaluation_ex.config
def contrastive_SCL():
    # change nprobe to 5
    device = "cuda"
    index_device = "cpu"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use GPU with optimization query id 4s"
    task_id = None
    query_id = 4
    nprobe = 5
    print(model_name)
    index_string = "IVF4096"
    save_path = "SCL-10"
    # targethitsfile = "/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargetdict.pkl"

    num_threads = 16
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    save_dir = (
        f"/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/{save_path}"  # IVF256
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    querysequences, targetsequences, all_hits = get_evaluation_data(
        query_id=query_id,
        save_dir=save_dir,
    )
    print("Loaded all data")
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
        "index_device": index_device,
        "index_string": index_string,  # IVF256,PQ16", #IVF256
        "nprobe": nprobe,
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 0,
        "max_seq_length": 512,
        "output_path": save_dir,
        "num_threads":num_threads,
    }
@evaluation_ex.config
def contrastive_alignments_quantized():
    # change nprobe to 5
    device = "cuda"
    index_device = "cpu"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveEvaluator"
    description = "Use GPU with optimization query id 4s"
    task_id = None
    query_id = 4
    nprobe = 5
    print(model_name)
    index_string = "IVF4096"
    save_path = "similarities-IVF"
    # targethitsfile = "/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargetdict.pkl"

    num_threads = 16
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)

    save_dir = (
        f"/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/{save_path}"  # IVF256
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    querysequences, targetsequences, all_hits = get_evaluation_data(
        query_id=query_id,
        save_dir=save_dir,
    )
    print("Loaded all data")
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
        "index_device": index_device,
        "index_string": index_string,  # IVF256,PQ16", #IVF256
        "nprobe": nprobe,
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 0,
        "max_seq_length": 2000,
        "output_path": save_dir,
        "num_threads":num_threads,
    }


#@evaluation_ex.config
def contrastive_alignments_kmer():
    device = "cuda"
    model_name = "ResNet1d"
    evaluator_name = "ContrastiveKmerEvaluator"
    description = "Use CPU with optimization"
    task_id = None
    testing = False
    print(task_id)
    print(model_name)

    num_threads = 6
    log_verbosity = logging.INFO
    root = f"{HOME}/prefilter/ResNet1d/23"
    checkpoint_path = f"{root}/checkpoints/best_epoch.ckpt"
    save_path = "similarities-IVF"

    with open(f"{root}/hparams.yaml", "r") as src:
        hparams = yaml.safe_load(src)
    # val_target_file = open("/xdisk/twheeler/daphnedemekas/target_data/evalfastanames.txt", "r")
    # val_targets = val_target_file.read().splitlines()
    save_dir = f"/xdisk/twheeler/daphnedemekas/prefilter-output/AlignmentEvaluation/{save_path}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    querysequences, targetsequences, all_hits = get_evaluation_data(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results", query_id=4, save_dir=save_dir,
    )
    print("Loaded all data")
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

    if testing:
        querysequences = dict(itertools.islice(querysequences.items(), 0, 100))
        targetsequences = dict(itertools.islice(targetsequences.items(), 0, 1000))

    print(f"Evaluating on {len(querysequences)} queries")
    evaluator_args = {
        "query_seqs": querysequences,
        "target_seqs": targetsequences,
        "hmmer_hits_max": all_hits,
        "encoding_func": None,
        "model_device": device,
        "index_device": "cpu",
        "index_string": "Flat",
        "figure_path": f"{ROOT}/AlignmentEvaluation/roc.png",
        "normalize_embeddings": True,
        "minimum_seq_length": 15,
        "max_seq_length": 512,
        "output_path": save_dir,
        "nprobe": 1,
    }
