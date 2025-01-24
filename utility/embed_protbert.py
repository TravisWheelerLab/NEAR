import torch
from transformers import BertModel, BertTokenizer
from Bio import SeqIO
import sys
import pickle
import numpy as np
import time


if len(sys.argv) != 3:
    print("Usage: python3 embed_fasta_protbert_aa.py <fasta path> <output path>")

# Load ProtBert model
model_name = "Rostlab/prot_bert"
print("Loading model", model_name)
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name)

# Transfer model to GPU if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device =", device)
model = model.to(device)
model.eval()

def embed_fasta(fasta_path, output_file, batch_size=32):
    # Load sequences from fasta file
    data = []
    data_order = {}
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        data.append((record.id, str(record.seq)))
        data_order[record.id] = str(i)

    # Process sequences in batches
    sequence_representations = {}
    start_time = time.time()
    print("Embedding sequences")
    for i in range(0, len(data), batch_size):
        if (i // batch_size) % 10 == 0:
            print(f"{i} of {len(data)} sequences processed")

        batch_data = data[i:i+batch_size]
        batch_lengths = [len(seq) for _, seq in batch_data]
        batch_tokens = tokenizer([" ".join(seq) for _, seq in batch_data], padding=True, return_tensors="pt", add_special_tokens=True)
        batch_tokens = {key: val.to(device) for key, val in batch_tokens.items()}

        # Compute embeddings
        with torch.no_grad():
            outputs = model(**batch_tokens, output_hidden_states=True)
            token_embeddings = outputs.hidden_states[-1]  # Get the last layer

        # Extract per-aa representations (mean over tokens)
        for j, (seq_id, _) in enumerate(batch_data):
            embeddings = token_embeddings[j][:batch_lengths[j]] #+ 2] # exclude special tokens
            sequence_representations[data_order[seq_id]] = embeddings.cpu().numpy()

    # Save embeddings to a file
    print("Embed time without file I/O = " + str(time.time() - start_time))
    np.savez(output_file, **{str(i): emb for i, emb in sequence_representations.items()})


fasta_path = sys.argv[1]
output_file = sys.argv[2]
embed_fasta(fasta_path, output_file)
print(f"Embeddings saved to {output_file}")