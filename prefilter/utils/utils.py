# pylint: disable=no-member
import json
import re
import os
import pdb
import logging
from random import shuffle, seed
from typing import Union, List, Tuple

import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils
from prefilter import MASK_FLAG, DECOY_FLAG
import prefilter.models as models

log = logging.getLogger(__name__)

seed(1)

__all__ = [
    "parse_labels",
    "create_faiss_index",
    "handle_figure_path",
    "fasta_from_file",
    "pad_contrastive_batches_with_labelvecs",
    "pad_contrastive_batches",
    "mask_mask",
    "load_model",
]


def load_model(model_path, hyperparams, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    state_dict = checkpoint["state_dict"]
    model = models.ResNet1d(**hyperparams).to(device)
    success = model.load_state_dict(state_dict)
    model.eval()
    return model, success


def create_faiss_index(embeddings, embed_dim, device="cpu", distance_metric="cosine"):
    print(f"using index with {distance_metric} metric.")

    if distance_metric == "cosine":
        index = faiss.IndexFlatIP(embed_dim)
    else:
        # transformer embeddings are _not_ normalized.
        index = faiss.IndexFlatL2(embed_dim)

    if device == "cuda":
        res = faiss.StandardGpuResources()
        # 0 is the index of the GPU.
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()

    index.add(embeddings)

    return index


def handle_figure_path(figure_path: str, ext: str = ".png") -> str:
    bs = os.path.basename(figure_path)
    name, curr_ext = os.path.splitext(bs)

    if len(curr_ext) == 0:
        figure_path = figure_path + ext

    return figure_path


def parse_labels(labelstring: str) -> Union[List[str], None]:
    """
    Parses the Pfam accession IDs from a > line in a fasta file.
    Assumes that the fasta files have been generated with prefilter.utils.label_fasta.
    Each > line of the fasta file should look like this:
    >arbitrary name of sequence | PFAMID1 PFAMID2 PFAMID3 ... PFAMIDN
    <sequence>
    or
    >arbitrary name of sequence | PFAMID1 (begin1, end1) PFAMID2 (begin2, end2) PFAMID3 ... PFAMIDN
    <sequence>
    Each sequence can have one or many pfam accession IDs as labels.
    If the fasta header doesn't have a | or it has a | followed by nothing list,
    :param labelstring: line to parse labels from
    :type labelstring: str
    :return: List of Pfam accession IDs
    :rtype: Union[List[str], None]
    """
    begin_char = labelstring.find("|")

    if begin_char == -1:
        return [DECOY_FLAG]

    if "(" in labelstring:
        # labelstring: ACC_ID (BEGIN END E_VALUE)
        labels = (
            labelstring[begin_char + 1 :].replace(")", "").replace("(", "").split(" ")
        )
        labels = list(filter(len, labels))
        labelset = []

        for i in range(0, len(labels), 4):
            accession_id, begin, end, e_value = (
                labels[i],
                labels[i + 1],
                labels[i + 2],
                labels[i + 3],
            )
            labelset.append([accession_id, begin, end, e_value])
        labels = labelset
    else:
        labels = labelstring[begin_char + 1 :].split(" ")

    labels = list(filter(len, labels))

    if not len(labels):
        return None

    return labels


def afa_from_file(afa_file: str):
    """
    Parse a .afa file.
    :param afa_file:
    :type afa_file:
    :return:
    :rtype:
    """
    labels, seqs = fasta_from_file(afa_file)
    return seqs


def fasta_from_file(fasta_file: str) -> Union[None, List[Tuple[str, str]]]:
    """
    Returns labels and sequences.
    param fasta_file: fasta file to load sequences + labels from.
    :type fasta_file: str
    :return: Labels, sequences, or none.
    :rtype: Union[None, List[List[str], List[str]]]
    """
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []

    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        sequence_strs.append("".join(buf))
        cur_seq_label = None
        buf = []

    with open(fasta_file, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:  # sequence line
                buf.append(line.strip())

    _flush_current_seq()

    return sequence_labels, sequence_strs


def _pad_sequences(sequences):
    mxlen = np.max([s.shape[-1] for s in sequences])
    padded_batch = np.zeros((len(sequences), LEN_PROTEIN_ALPHABET, mxlen))
    masks = []
    for i, s in enumerate(sequences):
        padded_batch[i, :, : s.shape[-1]] = s
        mask = np.ones((1, mxlen))
        mask[:, : s.shape[-1]] = 0
        masks.append(mask)

    masks = np.stack(masks)
    return torch.tensor(padded_batch).float(), torch.tensor(masks).bool()


def mask_mask(mask):
    idxs = torch.sum(~mask, axis=-1).squeeze().detach()
    for i, idx in enumerate(idxs):
        mask[i, :, (idx - 1) :] = True
    return mask


def pad_contrastive_batches(batch):
    member1 = [b[0] for b in batch]
    member2 = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    data = member1 + member2
    return (
        torch.stack(data),
        None,
        torch.as_tensor(labels),
    )


def pad_contrastive_batches_with_labelvecs(batch):
    """
    Pad batches that consist of a 3-tuple: seq, logo, and label
    :param batch: list of np.ndarrays encoding protein sequences/logos
    :type batch: List[np.ndarray]
    :return: torch.tensor
    :rtype: torch.tensor
    """

    pair1 = [b[0] for b in batch]
    pair2 = [b[1] for b in batch]
    lvec1 = [b[2] for b in batch]
    lvec2 = [b[3] for b in batch]
    data = pair1 + pair2
    labelvecs = lvec1 + lvec2
    return (
        torch.stack(data),
        None,
        labelvecs,
    )
