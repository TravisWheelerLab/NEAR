# pylint: disable=no-member
import json
import logging
import os
import pdb
import re
from random import seed, shuffle
from typing import List, Tuple, Union

import esm
import faiss
import faiss.contrib.torch_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.models as models

DECOY_FLAG = -1
MASK_FLAG = 1

log = logging.getLogger(__name__)

seed(1)

__all__ = [
    "parse_tblout",
    "esm_toks",
    "parse_labels",
    "create_faiss_index",
    "handle_figure_path",
    "fasta_from_file",
    "pad_contrastive_batches_with_labelvecs",
    "pad_contrastive_batches",
    "mask_mask",
    "load_model",
    "process_with_esm_batch_converter",
    "non_default_collate",
    "msa_transformer_collate",
    "daniel_sequence_encode",
]

TBLOUT_COL_NAMES = [
    "target_name",
    "query_name",
    "accession_id",
    "e_value",
    "description",
]
TBLOUT_COLS = [0, 2, 3, 4, 18]

esm_toks = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
]


def load_model(model_path, hyperparams, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    state_dict = checkpoint["state_dict"]
    if "training" in hyperparams:
        hyperparams["training"] = False
        if "apply_attention" not in hyperparams:
            hyperparams["apply_attention"] = False
        model = models.ResNet1d(**hyperparams).to(device)
    else:
        model = models.ResNet1d(**hyperparams, training=False).to(device)

    success = model.load_state_dict(state_dict)
    print(success)
    model.eval()
    success = 0

    return model, success


def create_faiss_index(
    embeddings, embed_dim, index_string, device="cpu", distance_metric="cosine"
):

    print(f"using index with {distance_metric} metric.")

    faiss.omp_set_num_threads(int(os.environ.get("NUM_THREADS")))

    k = int(10 * np.sqrt(embeddings.shape[0]).item())
    num_samples = 10 * k
    permutation = torch.randperm(embeddings.shape[0])
    embeds = embeddings[permutation[:num_samples]]
    if "IVF" in index_string:
        print(f"Following recs. on number of voronoi cells: {k}")
        index_string = index_string.format(k)

    print(f"Using index {index_string}")

    index = faiss.index_factory(embed_dim, index_string)

    if device == "cuda":
        num = 0
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(num), index)
        index.train(embeds)
        index.add(embeddings)
    else:
        index.train(embeds.to("cpu"))
        index.add(embeddings.to("cpu"))

    print("Done training index.")

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
    Assumes that the fasta files have been generated with src.utils.label_fasta.
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
    padded_batch = np.zeros((len(sequences), mxlen))
    masks = []
    for i, s in enumerate(sequences):
        padded_batch[i, : s.shape[-1]] = s
        mask = np.ones(mxlen)
        mask[: s.shape[-1]] = 0
        masks.append(mask)
    masks = np.stack(masks)
    return torch.as_tensor(padded_batch).float(), torch.as_tensor(masks).bool()


def mask_mask(mask):
    idxs = torch.sum(~mask, axis=-1).squeeze().detach()
    for i, idx in enumerate(idxs):
        mask[i, (idx - 1) :] = True
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


def non_default_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([torch.tensor(b[1]) for b in batch]),
        [b[2] for b in batch],
    )


def msa_transformer_collate(just_sequences=False, with_labelvectors=False):
    _, msa_transformer_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = msa_transformer_alphabet.get_batch_converter()

    def collate_fn(batch):
        if just_sequences:
            _, _, seq_embeddings = batch_converter([b[0] for b in batch])
            return (
                torch.as_tensor(seq_embeddings[:, :, 1:].squeeze()),
                [b[1] for b in batch],
                [b[2] for b in batch],
            )
        elif with_labelvectors:
            _, _, msa_embeds = batch_converter([b[0] for b in batch])
            _, _, seq_embeddings = batch_converter([[b[2]] for b in batch])
            # remove dummy dim and 0 begin-of-seq token.
            # seq_embeddings = seq_embeddings[:, :, 1:].squeeze()
            return (
                torch.as_tensor(msa_embeds),
                [torch.as_tensor(b[1]) for b in batch],
                torch.as_tensor(seq_embeddings),
                [torch.as_tensor(b[3]) for b in batch],
                [b[4] for b in batch],
            )

        else:
            _, _, msa_embeds = batch_converter([b[0] for b in batch])
            _, _, seq_embeddings = batch_converter([b[1] for b in batch])
            # remove dummy dim and 0 begin-of-seq token.
            seq_embeddings = seq_embeddings[:, :, 1:].squeeze()
            return (
                torch.as_tensor(msa_embeds),
                torch.as_tensor(seq_embeddings),
                [b[2] for b in batch],
            )

    return collate_fn


def daniel_sequence_encode(batch):
    seqs = []

    for seq in [b[0] for b in batch]:
        seqs.append(
            torch.stack(
                [models.amino_n_to_v[models.amino_a_to_n[s.upper()]] for s in seq]
            ).T
        )

    return (
        torch.cat([s.unsqueeze(0) for s in seqs]),
        [b[1] for b in batch],
        [b[2] for b in batch],
    )


def process_with_esm_batch_converter(return_alignments=False):
    _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    def process(batch):
        seqs = [("sd", "".join(b[0])) for b in batch]
        embeddings = [b[1] for b in batch]
        if return_alignments:
            return (
                batch_converter(seqs)[-1][:, 1:-1],
                [b[1] for b in batch],
                [b[2] for b in batch],
                [b[3] for b in batch],
            )

        else:
            return batch_converter(seqs)[-1][:, 1:-1], embeddings, [b[2] for b in batch]

    return process


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


def parse_tblout(tbl):
    """
    Parse a .tblout file created with hmmsearch -o <tbl>.tblout <seqdb> <hmmdb>
    :param tbl: .domtblout filename.
    :type tbl: str
    :return: dataframe containing the rows of the .tblout.
    :rtype: pd.DataFrame
    """

    if os.path.splitext(tbl)[1] != ".tblout":
        raise ValueError(f"must pass a .tblout file, found {tbl}")

    df = pd.read_csv(
        tbl,
        skiprows=3,
        header=None,
        delim_whitespace=True,
        usecols=TBLOUT_COLS,
        names=TBLOUT_COL_NAMES,
        engine="python",
        skipfooter=10,
    )

    df = df.dropna()

    # "-" is the empty label
    df["target_name"].loc[df["description"] != "-"] = (
        df["target_name"] + " " + df["description"]
    )

    return df
