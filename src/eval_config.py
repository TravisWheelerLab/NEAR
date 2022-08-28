import pdb
from glob import glob

import torch
from sacred import Experiment

from src.datasets.datasets import sanitize_sequence
from src.utils import char_to_index
from src.utils.helpers import non_default_collate

evaluation_ex = Experiment()


def wraps(device):
    def encode(sequence):
        seq = sanitize_sequence(sequence)
        seq = torch.as_tensor([char_to_index[c] for c in seq]).to(device)
        return seq

    return encode


# @evaluation_ex.config
# def config():
#
#     device = "cuda"
#     model_name = "ResNet1d"
#     evaluator_name = "AccuracyComputer"
#     model_path = "model_data/aug22/single_epoch_run/ResNet1d/1/"
#     checkpoint_path = (
#         "model_data/aug22/single_epoch_run/ResNet1d/1/checkpoints/epoch_0_2.174716.ckpt"
#     )
#
#     fasta_files = glob("/home/u4/colligan/data/prefilter/20piddata/train/*afa")
#
#     evaluator_args = {
#         "fasta_files": fasta_files,
#         "sequence_length": -1,
#         "include_all_families": False,
#         "n_seq_per_target_family": 1,
#         "normalize": True,
#         "embed_dim": 128,
#         "quantize_index": False,
#         "device": device,
#         "n_neighbors": 10,
#         "batch_size": 1,
#         "collate_fn": non_default_collate,
#     }


@evaluation_ex.config
def config():

    device = "cuda"
    index_device = "cuda"
    n_neighbors = 10
    quantize_index = True

    use_model_path = True

    model_name = "ResNet"
    evaluator_name = "UniRefEvaluator"
    model_path = "/home/u4/colligan/data/prefilter/cycle_16_2500.mod"

    checkpoint_path = (
        "model_data/aug22/single_epoch_run/ResNet1d/1/checkpoints/epoch_0_2.174716.ckpt"
    )

    evaluator_args = {
        "query_file": "/home/u4/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa",
        "target_file": "/home/u4/colligan/data/prefilter/uniref_benchmark/T_benchmark2k30k.fa",
        "normalize_embeddings": False,
        "encoding_func": wraps(device),
        "use_faiss": True,
        "quantize_index": quantize_index,
        "index_device": index_device,
        "n_neighbors": n_neighbors,
    }
