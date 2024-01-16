from src.utils.contrastive import _calc_embeddings
from src.utils.eval_utils import load_model
from src.data.hmmerhits import FastaFile
import time


# load model
checkpoint_path = ""
model_name = ""
device = "cuda"

model = load_model(checkpoint_path, model_name, device)

targetfasta = FastaFile("/xdisk/twheeler/daphnedemekas/prefilter/data/targets.fa")

sequences = list(targetfasta.values())

start = time.time()
embeddings = _calc_embeddings(sequences, model, device)
duration = time.time() - start

print(f"Embedding took: {duration}.")
print(f"Per query embedding: {duration / len(sequences)}.")
