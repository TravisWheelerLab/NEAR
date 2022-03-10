#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --job-name=label
#SBATCH --output=out/label-%A-%a.out
#SBATCH --array=[1-296]%50

names="/home/tc229954/data/prefilter/pfam/seed/model_comparison/training_data_no_evalue_threshold/200_file_subset_names.txt"
inp=$(sed -n "$SLURM_ARRAY_TASK_ID"p $names)
pfam="/home/tc229954/data/prefilter/pfam/seed/clustered/Pfam-A.0.5-train.hmm"

/home/tc229954/anaconda/envs/prefilter/bin/python prefilter/utils/label_fasta.py label /home/tc229954/data/prefilter/pfam/seed/clustered/0.5/$inp $pfam \
    -o /home/tc229954/max_hmmsearch/200_file_subset/