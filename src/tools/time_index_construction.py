import torch
import faiss
from src.utils.faiss_utils import create_faiss_index
import time
import numpy as np

embedding_file = "/xdisk/twheeler/daphnedemekas/prefilter/data/targets_embeddings.pt"
embeddings_file = "/xdisk/twheeler/daphnedemekas/NEAR-benchmarking/protbert/alltargets-masked.npy"
#embeddings = torch.load(embedding_file)
unrolled_targets = np.load(embeddings_file, allow_pickle=True)
unrolled_targets = np.concatenate(unrolled_targets)
print(unrolled_targets.shape)

#unrolled_targets = torch.cat(embeddings, dim=0)

index_device = "cuda"

start = time.time()

index = create_faiss_index(
    embeddings=unrolled_targets,
    embed_dim=unrolled_targets.shape[-1],
    distance_metric="cosine",
    index_string="IVF5000,PQ32",  # f"IVF{K},PQ8", #self.index_string, #f"IVF100,PQ8", #"IndexIVFFlat", #self.index_string,
    device=index_device,
)

duration = time.time() - start

print(f"Index Creation took: {duration}.")

faiss.write_index(index, "/xdisk/twheeler/daphnedemekas/NEAR-benchmarking/protbert/index-gpu.index")
