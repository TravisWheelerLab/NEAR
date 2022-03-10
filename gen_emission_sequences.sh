#!/usr/bin/env bash
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
n_seq=500

# usage: label_fasta.py inject [-h] [--relent RELENT]
#                              n fasta_files [fasta_files ...] output_directory
#                              ali_directory
#
# generate neighborhood emission sequences from the neighborhodd labels contained in
# "fasta_files"
#
# positional arguments:
#   n                 how many emission sequences to generate
#   fasta_files
#   output_directory  where to save the emitted sequences
#   ali_directory     where the .hmm files are saved
#
# optional arguments:
#   -h, --help        show this help message and exit
#   --relent RELENT   relative entropy to use when building hmms
set -e

for rel_ent in 0.45 0.5 0.55 0.65
do
  for subset in 200 1000
  do
  # only look at the subset with 1000 files since the 200 file subset is itself a subset of the 1000 file one
    $py_cmd /home/tc229954/share/prefilter/prefilter/utils/label_fasta.py inject $n_seq\
      ~/data/prefilter/pfam/seed/model_comparison/training_data/"$subset"_file_subset/*train*\
      ~/data/prefilter/pfam/seed/model_comparison/emission/"$rel_ent"_rel_ent/"$subset"_file_subset \
       ~/data/prefilter/pfam/seed/clustered/0.5/ --relent $rel_ent
  done
done
