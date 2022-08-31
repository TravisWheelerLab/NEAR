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


@evaluation_ex.config
def config():

    device = "cuda"
    index_device = "cpu"
    n_neighbors = 10
    distance_threshold = 0.5
    normalize_embeddings = False
    use_faiss = True
    hit_filename = "with_faiss.txt"
    istr = "OPQ64_128,IVF{}_HNSW32,PQ64"
    use_model_path = False
    target_file = "/home/u4/colligan/data/prefilter/uniprot_sprot.fasta"

    model_name = "ResNet"
    evaluator_name = "UniRefEvaluator"
    # TODO: Change the name of this directory.

    # checkpoint_path = (
    #     "model_data/aug22/single_epoch_run/ResNet1d/1/checkpoints/epoch_0_2.174716.ckpt"
    # )
    model_path = "/nsflj/tsdaf.txt"

    checkpoint_path = "/home/u4/colligan/data/prefilter/model_16.sdic"
    model_args = {
        "emb_dim": 256,
        "blocks": 5,
        "block_layers": 2,
        "first_kernel": 11,
        "kernel_size": 5,
        "groups": 2,
        "padding_mode": "reflect",
    }

    evaluator_args = {
        "query_file": "/home/u4/colligan/data/prefilter/uniref_benchmark/Q_benchmark2k30k.fa",
        "target_file": target_file,
        "normalize_embeddings": normalize_embeddings,
        "encoding_func": wraps(device),
        "use_faiss": use_faiss,
        "index_device": index_device,
        "index_string": istr,
        "n_neighbors": n_neighbors,
        "distance_threshold": distance_threshold,
        "hit_filename": hit_filename,
    }
