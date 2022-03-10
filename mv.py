from glob import glob
from shutil import copy
import os
import prefilter
import prefilter.utils as utils
import yaml

emission_files = glob("/home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/50_seq_per_family/*fa")
train_files = glob("/home/tc229954/data/prefilter/pfam/seed/model_comparison/training_data/1000_file_subset/*train*")

neighborhood_labels = []
for fasta_file in train_files:
    labels, _ = utils.fasta_from_file(fasta_file)
    for labelstring in labels:
        labelset = utils.parse_labels(labelstring)
        if len(labelset) > 1:
            for l in labelset[1:]:
                neighborhood_labels.append(l[0])

with open(prefilter.name_to_accession_id, 'r') as src:
    acc_id_to_name = {v: k for k, v in yaml.safe_load(src).items()}

for label in neighborhood_labels:
    emission_file = os.path.join("/home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/50_seq_per_family/",
                                 f"{acc_id_to_name[label]}_emission.fa")
    dst = f"/home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/subset/{acc_id_to_name[label]}_emission.fa"
    if os.path.isfile(emission_file) and not os.path.isfile(dst):
        copy(emission_file, dst)