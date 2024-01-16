import torch
import faiss
from src.utils.faiss_utils import create_faiss_index
import time

embedding_file = "/xdisk/twheeler/daphnedemekas/prefilter/data/targets_embeddings.pt"

embeddings = torch.load(embedding_file)
unrolled_targets = torch.cat(embeddings, dim=0)

index_device = "cuda"

start = time.time()

faiss.Index = create_faiss_index(
    embeddings=unrolled_targets,
    embed_dim=unrolled_targets.shape[-1],
    distance_metric="cosine",
    index_string="IVF5000,PQ32",  # f"IVF{K},PQ8", #self.index_string, #f"IVF100,PQ8", #"IndexIVFFlat", #self.index_string,
    device=index_device,
)

duration = time.time() - start

print(f"Index Creation took: {duration}.")
