import os
import pickle
import tqdm

def parse_mmseqs():
    # assert len(queries) == 16769
    #print(f"Number of relevant targets: {len(mytargets)}")
    outputdir = "/xdisk/twheeler/daphnedemekas/prefilter-output/mmseqs-reversed"

    # for subdir in tqdm.tqdm(os.listdir(directory)):

    mmseqspath = f"/xdisk/twheeler/daphnedemekas/mmseqs-rev/alnRes.m8"
    with open(mmseqspath, "r") as f:
        lines = f.readlines()
        for f in tqdm.tqdm(lines):
            line_split = f.split()
            query, target = line_split[0], line_split[1]
    #        if query not in myqueries or target not in mytargets:
    #            print("skipping")
    #            continue
            bitscore = float(line_split[-1])
            if os.path.exists(f"{outputdir}/{query}.txt"):
                with open(f"{outputdir}/{query}.txt", "a") as f:
                    f.write(f"{target}     {bitscore}" + "\n")
            else:
                with open(f"{outputdir}/{query}.txt", "w") as f:
                    f.write(f"{target}     {bitscore}" + "\n")


if __name__ == "__main__":
    parse_mmseqs()
