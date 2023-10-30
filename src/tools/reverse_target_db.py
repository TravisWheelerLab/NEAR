from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm

_, hmmer_hits = load_hmmer_hits(4)

fastafile = "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked.fa"
names_file = "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-names-masked.txt"
lengths_file = (
    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-lengths-masked.txt"
)

targetfasta = FastaFile(fastafile)

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

namefile = open(names_file, "w")

with open(
    lengths_file,
    "w",
) as f:
    for name, sequence in tqdm.tqdm(targetsequences.items()):
        # if name not in targets_that_are_hits:
        f.write(f">{name}\n{sequence[::-1]}\n")
        namefile.write(f"{name}\n")
        f.write(f"{len(sequence)}\n")

target_names = open(names_file, "r")
target_lengths = open(lengths_file, "r")
unrolled_names = []
for name, length in zip(target_names.readlines(), target_lengths.readlines()):
    unrolled_names.extend([name.strip("\n")] * int(length.strip("\n")))

with open("/xdisk/twheeler/daphnedemekas/unrolled-names-reversed-masked.txt", "w") as f:
    for name in unrolled_names:
        f.write(name + "\n")
