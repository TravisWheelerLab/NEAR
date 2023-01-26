# pylint: disable=no-member
import logging
import os
from random import seed
from typing import List, Tuple, Union

import faiss
import faiss.contrib.torch_utils
import pandas as pd
import torch

import src.models as models

DECOY_FLAG = -1
MASK_FLAG = 1

log = logging.getLogger(__name__)

seed(1)

__all__ = [
    "parse_tblout",
    "stack_vae_batch",
    "esm_toks",
    "parse_labels",
    "AAIndexFFT",
    "encode_with_aaindex",
    "pad_sequences",
    "create_faiss_index",
    "handle_figure_path",
    "fasta_from_file",
    "to_dict",
    "pad_contrastive_batches",
    "mask_mask",
    "stack_contrastive_batch",
    "load_model",
    "non_default_collate",
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


def to_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


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
    embeddings,
    embed_dim,
    index_string,
    nprobe,
    device="cpu",
    distance_metric="cosine",
):

    log.info(f"using index with {distance_metric} metric.")

    faiss.omp_set_num_threads(int(os.environ.get("NUM_THREADS")))

    if index_string == "Flat":
        if distance_metric == "cosine":
            log.info("Using normalized embeddings for cosine metric.")
            index = faiss.index_factory(
                embed_dim, index_string, faiss.METRIC_INNER_PRODUCT
            )
        else:
            index = faiss.index_factory(embed_dim, index_string)

        if device == "cuda":
            num = 0
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, int(num), index)

        log.info("Returning flat index without anything in it.")
        return index

    if "IVF" in index_string:
        embeddings = embeddings[torch.randperm(embeddings.shape[0])]
        log.info(f"Sampling {embeddings.shape[0]} embeddings.")

    log.info(f"Using index {index_string}")
    if "LSH" in index_string:
        index = faiss.IndexLSH(embed_dim, 64)
    else:
        if distance_metric == "cosine":
            log.info("Normalizing embeddings for use with cosine metric.")
            index = faiss.index_factory(
                embed_dim, index_string, faiss.METRIC_INNER_PRODUCT
            )
        else:
            index = faiss.index_factory(embed_dim, index_string)

        if "IVF" in index_string:
            index.nprobe = nprobe
            log.info(f"Setting nprobe to {nprobe}.")

    if device == "cuda":
        num = 0
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, int(num), index)
        index.train(embeddings.to("cuda"))
    else:
        index.train(embeddings.to("cpu"))

    log.info("Done training index.")

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
            labelstring[begin_char + 1 :]
            .replace(")", "")
            .replace("(", "")
            .split(" ")
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


def pad_sequences(sequences):
    mxlen = torch.max(torch.tensor([s.shape[-1] for s in sequences]))
    padded_batch = torch.zeros((len(sequences), sequences[0].shape[0], mxlen))
    masks = []
    for i, sequence in enumerate(sequences):
        padded_batch[i, :, : sequence.shape[-1]] = sequence
        mask = torch.ones(mxlen)
        mask[: sequence.shape[-1]] = 0
        masks.append(mask)
    masks = torch.stack(masks).unsqueeze(1)
    return torch.as_tensor(padded_batch).float(), torch.as_tensor(masks).bool()


def mask_mask(mask):
    idxs = torch.sum(~mask, axis=-1).squeeze().detach()
    for i, idx in enumerate(idxs):
        mask[i, (idx - 1) :] = True
    return mask


def pad_contrastive_batches_daniel(batch):
    member1 = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    return (
        torch.stack(member1),
        None,
        torch.as_tensor(labels),
    )


def pad_contrastive_batches(batch):
    member1 = [b[0] for b in batch]
    member2 = [b[1] for b in batch]
    labels = [b[2] for b in batch]

    return (
        torch.stack(member1),
        torch.stack(member2),
        torch.as_tensor(labels),
    )


def stack_vae_batch(batch):
    member1 = [b[0] for b in batch]
    member2 = [b[1] for b in batch]
    labels = [b[2] for b in batch]

    return (
        torch.stack(member1),
        torch.stack(member2),
        torch.as_tensor(labels),
    )


def stack_contrastive_batch(batch):
    member1 = [b[0] for b in batch]
    member2 = [b[1] for b in batch]
    labels = [b[2] for b in batch]

    return (
        torch.stack(member1 + member2),
        torch.as_tensor(labels),
    )


def non_default_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([torch.tensor(b[1]) for b in batch]),
        [b[2] for b in batch],
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


class AAIndexFFT:
    def __init__(self):
        self.mapping = {}

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __getitem__(self, protein):
        encoded = torch.fft.fft(
            torch.as_tensor([self.mapping[p] for p in protein])
        )
        return encoded


def encode_with_aaindex():

    with open("src/resources/indices.txt") as f:
        data = f.read()
    split = data.split("//")
    indices = [s[s.find("I") :].replace("\n", "").split() for s in split]

    aas = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

    indices = []

    for index in indices:
        if len(index) <= 20:
            logger.debug("Skipping index.")
            continue

        mapping = AAIndexFFT()
        broke = False
        for i, aa in enumerate(aas):
            try:
                mapping[aa] = float(index[-(20 - i)])
            except ValueError:
                broke = True
                break
        if not broke:
            indices.append(mapping)

    def enc(sequence):
        s1_fft = []
        for i in range(20):
            fft1 = torch.abs(indices[i][sequence])
            s1_fft.append(fft1)
        s1 = torch.stack(s1_fft, dim=0).squeeze()
        return s1

    return enc
