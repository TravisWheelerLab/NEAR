import os
from src.data.hmmerhits import FastaFile
import tqdm
import pdb

evaluation_targets_file = "/xdisk/twheeler/daphnedemekas/targetdataseqs/eval.txt"

print("Reading evaluation targets")
f = open(evaluation_targets_file)
evaltargets = [x.strip("\n") for x in f.readlines()]
f.close()

targetdatadir = "/xdisk/twheeler/daphnedemekas/prefilter/uniref/split_subset/targets"

evaltargetfile = open("/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargets.fa", "w")


for fastafile in tqdm.tqdm(os.listdir(targetdatadir)):
    fastadata = open(os.path.join(targetdatadir, fastafile), "r")
    fastadatatext = fastadata.read()
    fastadataentries = fastadatatext.split(">")
    # targetevalsequences = [s.split()[0] for s in fastadataentries[1:] if s in evaltargets]
    for entry in fastadataentries[1:]:
        sequence = entry.split()[0]
        if sequence in evaltargets:
            evaltargetfile.write(">" + entry)

evaltargetfile.close()
