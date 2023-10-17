from src.data.benchmarking import load_hmmer_hits
from src.data.hmmerhits import FastaFile


_, hmmer_hits = load_hmmer_hits(4)


targets_that_are_hits = set()

for targetlist in hmmer_hits.values():
    targetnames = list(targetlist.keys())
    targets_that_are_hits.update(targetnames)

targetfasta = FastaFile(f"data/targets-filtered.fa")

targetsequences = targetfasta.data

targets_to_reverse = []

with open("/xdisk/twheeler/daphnedemekas/data/reversedtargets-filtered.fa", "w") as f:
    for name, sequence in targetsequences.items():
        if name not in targets_that_are_hits:
            f.write(f">{name}\n{sequence}\n")
