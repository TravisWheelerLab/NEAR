"""
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger
(and a little bit Daniel)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


def load_t5_model(
    model_dir: Path = None,
    pretrained_model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
) -> tuple[T5EncoderModel, T5Tokenizer]:
    """
    Loads the T5 encoder model and its corresponding tokenizer.

    Args:
        model_dir (Path, optional): Path to a directory containing a cached model.
        pretrained_model_name (str): The name of the pretrained model to load.

    Returns:
        model (T5EncoderModel): The loaded T5 encoder model.
        tokenizer (T5Tokenizer): The corresponding tokenizer.
    """
    print(f"Loading model from: {pretrained_model_name}")

    if model_dir is not None:
        print("##########################")
        print(f"Loading cached model from: {model_dir}")
        print("##########################")

    model = T5EncoderModel.from_pretrained(pretrained_model_name, cache_dir=model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Cast model to full precision if running on CPU
    if device.type == "cpu":
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name, do_lower_case=False)

    print(f"Using device: {device}")
    print("Model and tokenizer loaded.")
    return model, tokenizer


def read_fasta_file(fasta_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    """
    Reads in a FASTA file containing protein sequences and returns a dictionary
    of {identifier: sequence} pairs and an ordering dictionary.

    Args:
        fasta_path (Path): Path to the FASTA file.

    Returns:
        sequences (dict[str, str]): Mapping of sequence identifier to sequence.
        seq_order (dict[str, int]): Mapping of sequence identifier to its order.
    """
    sequences = {}
    seq_order = {}
    print(f"Reading FASTA from: {fasta_path}")

    with open(fasta_path, "r") as f:
        current_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Grab the identifier from the header
                uniprot_id = line.replace(">", "").strip()
                seq_order[uniprot_id] = len(seq_order)
                uniprot_id = (
                    uniprot_id.replace("/", "_")
                    .replace(".", "_")
                )
                sequences[uniprot_id] = ""
                current_id = uniprot_id
            else:
                # Merge lines into one continuous sequence, removing gaps
                if current_id is not None:
                    sequences[current_id] += (
                        "".join(line.split()).upper().replace("-", "")
                    )

    return sequences, seq_order


def get_embeddings(
    seq_path: Path,
    emb_path: Path,
    model_dir: Path,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch: int = 100,
) -> bool:
    """
    Computes T5 embeddings for sequences in a FASTA file and saves them to a NumPy .npz file.

    Args:
        seq_path (Path): Path to the FASTA file containing input protein sequences.
        emb_path (Path): Path to the output .npz file for saving embeddings.
        model_dir (Path): Path to a directory holding a cached model, if any.
        max_residues (int, optional): Maximum number of cumulative residues in a batch.
        max_seq_len (int, optional): Length threshold above which sequences are
            processed in single-sequence batches to avoid OOM errors.
        max_batch (int, optional): Maximum number of sequences per batch.

    Returns:
        bool: True if embeddings are successfully computed and saved, otherwise False.
    """
    sequences, seq_order = read_fasta_file(seq_path)
    model, tokenizer = load_t5_model(model_dir)

    # Summaries
    print("########################################")
    example_id = next(iter(sequences))
    print(f"Example sequence: {example_id}\n{sequences[example_id]}")
    print("########################################")
    print(f"Total number of sequences: {len(sequences)}")

    avg_length = sum(len(seq) for seq in sequences.values()) / len(sequences)
    n_long = sum(1 for seq in sequences.values() if len(seq) > max_seq_len)

    # Sort sequences by descending length
    sequences_sorted = sorted(sequences.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"Average sequence length: {avg_length:.2f}")
    print(f"Number of sequences > {max_seq_len}: {n_long}")

    start_time = time.time()
    emb_dict = {}
    batch = []

    device = next(model.parameters()).device

    for seq_idx, (seq_id, seq) in enumerate(sequences_sorted, start=1):
        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        seq_len = len(seq)
        spaced_seq = " ".join(list(seq))
        batch.append((seq_id, spaced_seq, seq_len))

        # Estimate how many residues will be in the batch after adding this seq
        total_batch_res = sum(s_len for _, _, s_len in batch) + seq_len

        # Conditions to process the current batch:
        # 1) batch is full (>= max_batch)
        # 2) batch would exceed max_residues
        # 3) last sequence overall
        # 4) single very long sequence (seq_len > max_seq_len)
        if (
            len(batch) >= max_batch
            or total_batch_res >= max_residues
            or seq_idx == len(sequences_sorted)
            or seq_len > max_seq_len
        ):
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch.clear()

            # Tokenize
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding["input_ids"], device=device)
            attention_mask = torch.tensor(token_encoding["attention_mask"], device=device)

            # Generate embeddings
            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print(
                    f"RuntimeError during embedding for {seq_id} (length={seq_len}). "
                    "Try lowering batch size or sequence length. If this also fails for single-sequence "
                    "batches, more VRAM is required for your protein."
                )
                continue

            # Process each sequence in the batch
            for b_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[b_idx]
                emb = embedding_repr.last_hidden_state[b_idx, :s_len]

                # For the first one, print shape info
                if not emb_dict:
                    print(
                        f"Embedded protein {identifier} with length {s_len} "
                        f"to embedding of shape: {emb.shape}"
                    )

                emb_dict[identifier] = emb.half().cpu().numpy().squeeze()

    end_time = time.time()
    embed_time = end_time - start_time
    print(f"Embedding time = {embed_time:.2f} seconds")

    # Save embeddings to npz, using the original order's integer indices as keys
    np.savez(emb_path, **{str(seq_order[key]): emb_dict[key] for key in emb_dict})

    print("\n############# STATS #############")
    print(f"Total number of embeddings: {len(emb_dict)}")
    print(
        f"Time: {embed_time:.2f} [s] | Time/protein: {embed_time / max(len(emb_dict),1):.4f} [s] | "
        f"Average length = {avg_length:.2f}"
    )
    print(f"Total time: {embed_time:.2f} seconds")
    return True


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Creates and returns the ArgumentParser object for command-line arguments.

    Returns:
        parser (ArgumentParser): The argument parser for this script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Creates T5 embeddings for a given FASTA file containing protein sequence(s)."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to a FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path (filename.npz) to save the embeddings in NumPy npz format.",
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default=None,
        help="Path to a directory with a cached pre-trained T5 model (optional).",
    )
    
    return parser


def main() -> None:
    """
    Main execution function that parses arguments and triggers the embedding process.
    """
    parser = create_arg_parser()
    args = parser.parse_args()

    seq_path = Path(args.input)
    emb_path = Path(args.output)
    model_dir = Path(args.model) if args.model else None

    get_embeddings(seq_path, emb_path, model_dir)


if __name__ == "__main__":
    main()
