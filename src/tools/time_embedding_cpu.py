from src.evaluator.contrastive import _calc_embeddings
from src.utils.eval_utils import load_model
from src.data.hmmerhits import FastaFile
from multiprocessing.pool import ThreadPool as Pool
import time


model_name = "ResNet1d"
device = "cpu"
checkpoint_path = "data/best_epoch.ckpt"
model_name = "ResNet1d"
model = load_model(checkpoint_path, model_name, device)
targetfasta = FastaFile("data/targets.fa")

sequences = list(targetfasta.data.values())


pool = Pool(16)

start = time.time()
idx = 0

arg_list = [
    sequences,
    model,
]

for result in pool.imap(_calc_embeddings, arg_list):
    print(f"Finished thread: {idx}")

pool.terminate()
duration = time.time() - start

print(f"NEAR embedding duration on CPU parallelized: {duration}")

print(f"Per query embedding: {duration / len(sequences)}")
