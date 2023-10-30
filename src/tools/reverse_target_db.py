from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm

_, hmmer_hits = load_hmmer_hits(4)

fastafile = "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked.fa"
names_file = "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-names-masked.txt"
lengths_file = (
    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-lengths-masked.txt"
)
reversed_fasta_file = (
    "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked-reversed.fa"
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
fasta = open(reversed_fasta_file, "w")
lengthsfile = open(lengths_file, "w")
for name, sequence in tqdm.tqdm(targetsequences.items()):
    # if name not in targets_that_are_hits:
    fasta.write(f">{name}\n{sequence[::-1]}\n")
    namefile.write(f"{name}\n")
    lengthsfile.write(f"{len(sequence)}\n")
lengthsfile.close()
namefile.close()
fasta.close()

target_names = open(names_file, "r")
target_lengths = open(lengths_file, "r")
unrolled_names = []
unrolled_lengths = []

for name, length in zip(target_names.readlines(), target_lengths.readlines()):
    unrolled_names.extend([name.strip("\n")] * int(length.strip("\n")))
    unrolled_lengths.extend([int(length.strip("\n"))] * int(length.strip("\n")))

with open(
    "/xdisk/twheeler/daphnedemekas/unrolled-lengths-masked-reversed.txt", "w"
) as f:
    for length in unrolled_lengths:
        f.write(str(length) + "\n")

with open("/xdisk/twheeler/daphnedemekas/unrolled-names-masked-reversed.txt", "w") as f:
    for name in unrolled_names:
        f.write(str(name) + "\n")
