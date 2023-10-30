from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm

_, hmmer_hits = load_hmmer_hits(4)
targetfasta = FastaFile(
    f"/xdisk/twheeler/daphnedemekas/prefilter/data/targets-filtered-masked.fa"
)

targetsequences = targetfasta.data

targets_that_are_hits = set()

for targetlist in tqdm.tqdm(hmmer_hits.values()):
    targetnames = list(targetlist.keys())
    for target in targetnames:
        if target in targetsequences:
            targets_that_are_hits.add(target)

print(f"Number of targets that are hits: {len(targetnames)}")


print(f"Number of target sequences: {len(targetsequences)}")

targets_to_reverse = []

names = []
lengths = []

namefile = open(
    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-names-masked.txt", "w"
)
lengthsfile = open(
    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-lengths-masked.txt", "w"
)
with open(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/reversedtargets-masked.fa",
    "w",
) as f:
    for name, sequence in tqdm.tqdm(targetsequences.items()):
        # if name not in targets_that_are_hits:
        f.write(f">{name}\n{sequence[::-1]}\n")
        namefile.write(f"{name}\n")
        lengthsfile.write(f"{len(sequence)}\n")
