import torch
import esm
import numpy as np
from Bio import SeqIO
import sys
import pickle
import time
if len(sys.argv) != 3:
    print("Usage: python3 embed_fasta_esm.py <fasta path> <output path>")

# Load the ESM-2 model
model_name = "esm2_t36_3B_UR50D"
print("Loading model", model_name)
try:
    # Try loading the model locally first
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_name)
except FileNotFoundError:
    # If not found, download the model
    print(f"Model {model_name} not found locally. Downloading from hub...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

# Transfer model to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =",device)
model = model.to(device)

def embed_fasta(fasta_path, output_file, batch_size=32):
    # Load sequences from fasta file
    data = []
    data_order = {}
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        data.append((record.id, str(record.seq).upper()))
        data_order[record.id] = str(i)

    # Prepare batch converter
    batch_converter = alphabet.get_batch_converter()

    # Process sequences in batches
    sequence_representations = {}
    c = 0
    print("Embedding sequences")
    start_time = time.time()
    for i in range(0, len(data), batch_size):
        if (i // batch_size) % 32 == 0:
            print(i, "of", len(data))
        batch_data = data[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        batch_lengths = [len(s[1]) for s in batch_data]

        # Compute embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])  # Get the last layer representations
            token_embeddings = results["representations"][33]

        # Extract per-sequence representations (mean over tokens)
        for j, (seq_id, _) in enumerate(batch_data):
            sequence_rep = token_embeddings[j, :batch_lengths[j]].cpu().numpy()
            sequence_representations[data_order[seq_id]] = sequence_rep

    print("embedding time without file I/O = " + str(time.time() - start_time))
    # Save embeddings to a npy file
    np.savez(output_file, **{str(i): emb for i, emb in sequence_representations.items()})



fasta_path = sys.argv[1]
output_file = sys.argv[2]
embed_fasta(fasta_path, output_file)
print(f"Embeddings saved to {output_file}")
