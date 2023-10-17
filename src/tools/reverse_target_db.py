from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm

_, hmmer_hits = load_hmmer_hits(4)


targets_that_are_hits = set()

for targetlist in hmmer_hits.values():
    targetnames = list(targetlist.keys())
    targets_that_are_hits.update(targetnames)

targetfasta = FastaFile(
    f"/xdisk/twheeler/daphnedemekas/prefilter/data/targets-filtered.fa"
)

targetsequences = targetfasta.data

print(f"Number of target sequences: {len(targetsequences)}")

targets_to_reverse = []

with open(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/reversedtargets-filtered.fa", "w"
) as f:
    for name, sequence in tqdm.tqdm(targetsequences.items()):
        if name not in targets_that_are_hits:
            f.write(f">{name}\n{sequence}\n")
