from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm
import pickle

hmmer_max_hits, hmmer_normal_hits = load_hmmer_hits(4)
targetfasta = FastaFile(
    f"/xdisk/twheeler/daphnedemekas/prefilter/data/targets-filtered.fa"
)

targetsequences = targetfasta.data

targets_that_are_hits = set()

hmmer_eval_hits_max = {}
hmmer_eval_hits_normal = {}

for query, targetlist in tqdm.tqdm(hmmer_max_hits.items()):
    targetnames = list(targetlist.keys())
    for target in targetnames:
        if target in targetsequences:
            if query not in hmmer_eval_hits_max:
                hmmer_eval_hits_max[query] = []
            hmmer_eval_hits_max[query].append(target)


with open(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/evaltargetdictmax.pkl", "wb"
) as f:
    pickle.dump(hmmer_eval_hits_max, f)


for query, targetlist in tqdm.tqdm(hmmer_normal_hits.items()):
    targetnames = list(targetlist.keys())
    for target in targetnames:
        if target in targetsequences:
            if query not in hmmer_eval_hits_max:
                hmmer_eval_hits_normal[query] = []
            hmmer_eval_hits_normal[query].append(target)
with open(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/evaltargetdictnormal.pkl", "wb"
) as f:
    pickle.dump(hmmer_eval_hits_normal, f)
