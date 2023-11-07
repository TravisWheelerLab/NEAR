from src.data.hmmerhits import FastaFile
import pickle


def get_lengths(targetfasta, queryfasta):
    targetlengths = {}
    querylengths = {}
    targetlengthsfile = (
        "/xdisk/twheeler/daphnedemekas/prefilter/data/target-lengths.pkl"
    )
    querylengthsfile = "/xdisk/twheeler/daphnedemekas/prefilter/data/query-lengths.pkl"

    for name, sequence in targetfasta.data.items():
        targetlengths[name] = len(sequence)
    for name, sequence in queryfasta.data.items():
        querylengths[name] = len(sequence)
    with open(targetlengthsfile, "wb") as f:
        pickle.dump(targetlengths, f)
    with open(querylengthsfile, "wb") as f:
        pickle.dump(targetlengths, f)


def get_lengths_masked(targetfastamasked, queryfastamasked):
    targetlengths = {}
    querylengths = {}
    targetlengthsfile = (
        "/xdisk/twheeler/daphnedemekas/prefilter/data/target-lengths-masked.pkl"
    )
    querylengthsfile = (
        "/xdisk/twheeler/daphnedemekas/prefilter/data/query-lengths-masked.pkl"
    )

    for name, sequence in targetfastamasked.data.items():
        newseq = sequence.replace("X", "")
        targetlengths[name] = len(newseq)
    for name, sequence in queryfasta.data.items():
        newseq = sequence.replace("X", "")
        querylengths[name] = len(newseq)
    with open(targetlengthsfile, "wb") as f:
        pickle.dump(targetlengths, f)
    with open(querylengthsfile, "wb") as f:
        pickle.dump(targetlengths, f)


targetfasta = FastaFile("/xdisk/twheeler/daphnedemekas/prefilter/data/targets.fa")
queryfasta = FastaFile(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/queries-filtered.fa"
)

targetfastamasked = FastaFile(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-masked.fa"
)

queryfastamasked = FastaFile(
    "/xdisk/twheeler/daphnedemekas/prefilter/data/queries-masked.fa"
)
get_lengths(targetfasta, queryfasta)
get_lengths_masked(targetfastamasked, queryfastamasked)
