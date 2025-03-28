import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio import SeqIO

from esm.models.esmc import ESMC


def sequence_tensors_by_length(file_path: str, model: ESMC) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[int]]]:
    """
    Reads a FASTA file and tokenizes its sequences by length.

    Args:
        file_path (str): Path to the FASTA file.
        model (ESMC): The ESM model (with a `_tokenize` method).

    Returns:
        length_to_seqs (Dict[int, torch.Tensor]): Mapping from sequence length to
            a batched tensor of tokenized sequences of that length.
        length_to_ids (Dict[int, List[int]]): Mapping from sequence length to a list
            of sequence indices (in the order they appeared in the FASTA file).
    """
    length_to_seqs = {}
    length_to_ids = {}

    for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
        seq_length = len(record.seq)

        if seq_length not in length_to_seqs:
            length_to_seqs[seq_length] = []
            length_to_ids[seq_length] = []

        length_to_seqs[seq_length].append(str(record.seq).upper())
        length_to_ids[seq_length].append(i)

    # Tokenize sequences for each length
    for length in length_to_seqs.keys():
        length_to_seqs[length] = model._tokenize(length_to_seqs[length])

    return length_to_seqs, length_to_ids


def embed_sequence_tensors(
    model: ESMC,
    sequence_tensors: Dict[int, torch.Tensor],
    device: torch.device,
    mean_embeddings: bool,
    batch_size: int = 512,
) -> Dict[int, np.ndarray]:
    """
    Generates embeddings for tokenized sequences, optionally taking the mean
    across positions. Handles batching to avoid memory issues.

    Args:
        model (ESMC): The ESM model to use for generating embeddings.
        sequence_tensors (Dict[int, torch.Tensor]): A mapping from sequence length
            to a batched tensor of tokenized sequences.
        device (torch.device): The device (CPU or GPU) to use.
        mean_embeddings (bool): If True, average embeddings over the sequence dimension.
        batch_size (int): Batch size for processing embeddings.

    Returns:
        embeddings (Dict[int, np.ndarray]): A mapping from sequence length to an array
            of embeddings. Dimensions depend on whether `mean_embeddings` is used.
    """
    embeddings = {}

    for length, seq_tensor in sequence_tensors.items():
        with torch.no_grad():
            if len(seq_tensor) <= batch_size:
                output = model(seq_tensor.to(device)).embeddings[:, 1:-1, :]
                if mean_embeddings:
                    output = output.mean(dim=1, keepdim=True)
                embeddings[length] = output
            else:
                # Process in smaller batches
                num_sequences = seq_tensor.size(0)
                num_batches = (num_sequences + batch_size - 1) // batch_size
                batch_embeddings = []

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_sequences)
                    batch = seq_tensor[start_idx:end_idx].to(device)

                    out = model(batch).embeddings[:, 1:-1, :]
                    if mean_embeddings:
                        out = out.mean(dim=1, keepdim=True)
                    batch_embeddings.append(out.cpu())

                embeddings[length] = torch.cat(batch_embeddings, dim=0)

        # Convert to half precision on CPU and then to numpy for storage
        embeddings[length] = embeddings[length].half().cpu().numpy()

    return embeddings


def save_embeddings(
    length_to_embeddings: Dict[int, np.ndarray],
    length_to_ids: Dict[int, List[int]],
    output_path: str,
) -> None:
    """
    Saves the embeddings in NumPy NPZ format. Each key in the NPZ file is the
    sequence index, and each value is its corresponding embedding.

    Args:
        length_to_embeddings (Dict[int, np.ndarray]): A mapping from sequence length
            to an array of embeddings for that length.
        length_to_ids (Dict[int, List[int]]): A mapping from sequence length to the
            list of sequence indices in the order they appeared in the FASTA file.
        output_path (str): Path to the output file (e.g., 'embeddings.npz').
    """
    embedding_dict = {}
    for length, ids in length_to_ids.items():
        for i, seq_id in enumerate(ids):
            embedding_dict[str(seq_id)] = length_to_embeddings[length][i]

    np.savez(output_path, **embedding_dict)


def main():
    """
    Main function that loads a model, reads a FASTA file, computes embeddings,
    and saves them to disk.
    """
    if not (4 <= len(sys.argv) <= 5):
        print("Usage: python3 esmc_embed.py <esmc_300m/esmc_600m> <fasta path> <output path> [mean]")
        sys.exit(1)

    model_name = sys.argv[1]
    fasta_path = sys.argv[2]
    output_file = sys.argv[3]
    mean_embeddings = (len(sys.argv) == 5 and sys.argv[4] == "mean")

    global_start_time = time.time()

    model = ESMC.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Device: {device}")
    if mean_embeddings:
        print("Using mean embeddings across sequence dimension.")

    length_to_seqtensor, length_to_ids = sequence_tensors_by_length(fasta_path, model)

    print("Embedding tensors...")
    start_time = time.time()
    embeddings = embed_sequence_tensors(model, length_to_seqtensor, device, mean_embeddings)
    end_time = time.time()

    print(f"Done. Embedding time (s): {end_time - start_time}")
    print("Saving embeddings...")
    save_embeddings(embeddings, length_to_ids, output_file)

    global_end_time = time.time()
    print(f"Done! Total time: {global_end_time - global_start_time}")


if __name__ == "__main__":
    main()
