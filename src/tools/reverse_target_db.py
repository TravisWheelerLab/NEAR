from src.data.compare_models import load_hmmer_hits
from src.data.hmmerhits import FastaFile
import tqdm


def reverse(sequences, save_dir):
    with open(
        save_dir,
        "w",
    ) as f:
        for name, sequence in tqdm.tqdm(sequences.items()):
            f.write(f">{name}\n{sequence[::-1]}\n")
    # check
    reversedfasta = FastaFile(save_dir)

    reversedseqs = reversedfasta.data

    assert len(sequences) == len(reversedseqs)
    assert list(sequences.keys()) == list(reversedseqs.keys())


targetfasta = FastaFile(
    f"/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked.fa"
)

targetsequences = targetfasta.data

targets_reversed = (
    "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked-reversed.fa"
)


# targets_that_are_hits = set()
# names = []
# lengths = []
#
# namefile = open(
#    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-names-masked.txt", "w"
# )
# lengthsfile = open(
#    "/xdisk/twheeler/daphnedemekas/prefilter/reversed-target-lengths-masked.txt", "w"
# )

targets_reversed = (
    "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked-reversed.fa"
)
