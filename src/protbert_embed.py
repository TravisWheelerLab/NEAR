import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
from Bio import SeqIO
from transformers import BertModel, BertTokenizer


def embed_and_save_fasta(
    fasta_path: str,
    output_file: str,
    mean_embeddings: bool = False,
    batch_size: int = 32,
) -> None:
    """
    Embeds protein sequences from a FASTA file using the ProtBert model and saves
    the embeddings to a NumPy NPZ file.

    Args:
        fasta_path (str): Path to the input FASTA file.
        output_file (str): Path to the output NPZ file where embeddings will be saved.
        mean_embeddings (bool, optional): If True, computes the mean embedding across
            all residues for each sequence. Defaults to False (per-residue embeddings).
        batch_size (int, optional): Number of sequences to process in each batch.
            Defaults to 32.
    """
    # Load sequences from FASTA file
    sequences = []
    seq_index_map: Dict[str, str] = {}
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        sequences.append((record.id, str(record.seq)))
        seq_index_map[record.id] = str(i)

    print(f"Total number of sequences: {len(sequences)}")
    print(f"Embedding sequences in batches of {batch_size}...")

    # Initialize representations dictionary
    sequence_representations = {}

    # Time the embedding process (excluding file I/O)
    start_time = time.time()

    # Process sequences in batches
    for i in range(0, len(sequences), batch_size):
        batch_data = sequences[i : i + batch_size]
        batch_lengths = [len(seq) for _, seq in batch_data]

        # Tokenize (add spaces between AAs for ProtBert) and batch-encode
        inputs = [" ".join(seq) for _, seq in batch_data]
        batch_tokens = tokenizer(
            inputs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        batch_tokens = {key: val.to(device) for key, val in batch_tokens.items()}

        # Compute embeddings
        with torch.no_grad():
            outputs = model(**batch_tokens, output_hidden_states=True)
            token_embeddings = outputs.hidden_states[-1]  # Last hidden layer

        # Extract per-residue or mean embeddings
        for j, (seq_id, _) in enumerate(batch_data):
            seq_len = batch_lengths[j]
            # Skip special tokens; take just the length of the sequence
            embeddings = token_embeddings[j][1 : seq_len + 1]

            # Optionally compute mean embedding
            if mean_embeddings:
                embeddings = embeddings.mean(dim=0, keepdim=True)

            # Store embeddings (in half precision)
            sequence_representations[seq_index_map[seq_id]] = embeddings.cpu().half().numpy()

    # Print time spent on embeddings (excluding file save)
    print(f"Embedding time (no file I/O): {time.time() - start_time:.2f}s")

    # Save embeddings to .npz file
    np.savez(output_file, **sequence_representations)
    print(f"Embeddings saved to: {output_file}")


def main() -> None:
    """
    Main function to parse command-line arguments, load ProtBert,
    and generate embeddings for sequences in a FASTA file.
    """
    start_global_time = time.time()

    # Check command-line arguments
    if len(sys.argv) not in [3, 4]:
        print("Usage: python protbert_embed.py <FASTA_PATH> <OUTPUT_PATH>")
        sys.exit(1)

    fasta_path = sys.argv[1]
    output_file = sys.argv[2]
    mean_embeddings = False

    # Optional mean flag
    if len(sys.argv) == 4 and sys.argv[3] == "mean":
        mean_embeddings = True
        print("Using mean embeddings per protein.")

    # Load model and tokenizer
    global model, tokenizer, device
    model_name = "Rostlab/prot_bert"
    print(f"Loading model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model.to(device)
    model.eval()

    # Perform embedding
    embed_and_save_fasta(fasta_path, output_file, mean_embeddings)

    # Total execution time
    print(f"Total runtime: {time.time() - start_global_time:.2f}s")


if __name__ == "__main__":
    main()
