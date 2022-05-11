import os
import pdb
import pandas as pd

from glob import glob
import prefilter.utils as utils

true_valid = glob(
    "/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_sequences/*fa"
)
tblout_path = "/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_tblouts/"
out_path = "/home/tc229954/data/prefilter/pfam/seed/20piddata/valid_set_subsampled_by_what_hmmer_gets_right/"

n_seq = 0
n_correct = 0

for file in true_valid:
    # grab the sequences in the split validation set
    headers, seqs = utils.fasta_from_file(file)
    tblout_file = os.path.join(tblout_path, os.path.basename(file) + ".tblout")
    tblout_df = utils.parse_tblout(tblout_file)
    # grab the filename;
    correct_target = os.path.basename(file).replace(".afa.fa", "")
    if "-2.afa" in os.path.basename(file):
        correct_target = correct_target.replace("-2", "-1")
    else:
        correct_target = correct_target.replace("-1", "-2")
    # now, remove entries from the df that don't contain the correct str
    # the correct str is the matching split fasta; swap the 1 and 2 in the valid file.
    # i need to change this in the rust code.
    if not tblout_df.shape[0]:
        print(f"no hits for {file}.")
        continue
    tblout_df = tblout_df.loc[tblout_df["query_name"].str.contains(correct_target), :]
    # now, grab the sequences from the validation file that are in the
    # correctly classified set;
    correctly_classified_names = set(tblout_df["target_name"])
    if len(set(headers).intersection(correctly_classified_names)):
        with open(os.path.join(out_path, os.path.basename(file)), "w") as dst:
            for header, seq in zip(headers, seqs):
                if header in correctly_classified_names:
                    # write the sequence and the header to the file
                    dst.write(f">{header}\n{seq}\n")
    else:
        print(f"No correct hits for {file}")
