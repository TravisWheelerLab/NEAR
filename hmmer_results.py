import os
import pdb

from glob import glob

true_valid = glob("/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_sequences/names/*hits")
hit_valid = glob("/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_tblouts/names/*hits")


# parse each file in the sets;
n_seq = 0
n_correct = 0

for file in true_valid:
    valid_file = file.replace("valid_sequences", "valid_tblouts")
    valid_file = valid_file.replace("true", "tblout")
    if not os.path.isfile(valid_file):
        print(f"Couldn't find {valid_file}")
        continue

    with open(file, "r") as src:
        true_names = src.readlines()
        true_names = [t.replace("\n", "").replace(">", "") for t in true_names]

    with open(valid_file, "r") as src:
        valid_names = src.readlines()
        valid_names = [t.replace("\n", "") for t in valid_names]
    true_names = [t[:t.find(" ")] for t in true_names]
    valid_names = [t[:t.find(" ")] for t in valid_names]

    for name in true_names:
        n_seq += 1
        if name in valid_names:
            n_correct += 1

print(n_seq, n_correct, n_correct / n_seq)
