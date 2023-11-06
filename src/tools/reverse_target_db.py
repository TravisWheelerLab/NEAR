from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm

#_, hmmer_hits = load_hmmer_hits(4)
targetfasta = FastaFile(
    f"/xdisk/twheeler/daphnedemekas/prefilter/data/targets.fa"
)

targetsequences = targetfasta.data

#targets_that_are_hits = set()
#names = []
#lengths = []
#
#namefile = open(
#    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-names-masked.txt", "w"
#)
#lengthsfile = open(
#    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-lengths-masked.txt", "w"
#)
with open(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-reversed.fa",
    "w",
) as f:
    for name, sequence in tqdm.tqdm(targetsequences.items()):
        f.write(f">{name}\n{sequence[::-1]}\n")
       # namefile.write(f"{name}\n")
       #     lengthsfile.write(f"{len(sequence)}\n")

reversedfasta = FastaFile(f"/xdisk/twheeler/daphnedemekas/prefilter/data/targets.fa")

reversedseqs = reversedfasta.data

assert len(targetsequences) == len(reversedseqs)
assert list(targetsequences.keys()) == list(reversedseqs.keys())
