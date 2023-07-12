import tqdm
import pdb

targetdatafasta = "/xdisk/twheeler/daphnedemekas/prefilter/data/targets-filtered.fa"

shuffledfasta = "/xdisk/twheeler/daphnedemekas/prefilter/data/reversedtargets.fa"

with open(targetdatafasta, "r") as targets:
    alltext = targets.read()

seqs = alltext.split(">")


writefile = open(shuffledfasta, "w")
for seq in tqdm.tqdm(seqs[1:]):
    if len(seq.split("\n")) != 3:
        print(f"Error: {seq}")
        continue
    name, sequence, _ = seq.split("\n")
    if len(_) > 0:
        pdb.set_trace()
    reversed_seq = sequence[::-1]
    writefile.write(f">{name}" + "\n")
    writefile.write(reversed_seq + "\n")

writefile.close()
