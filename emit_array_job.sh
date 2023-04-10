#!/usr/bin/env bash
#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --job-name=label
#SBATCH --output=label.out
#SBATCH --array=[1-18820]%200

emission_name=$(sed -n "$SLURM_ARRAY_TASK_ID"p emission_names.txt)
echo "$emission_name"
/home/tc229954/anaconda/envs/prefilter/bin/python \
/home/tc229954/share/prefilter/prefilter/utils/label_fasta.py label $emission_name /home/tc229954/subset/0.5/Pfam-0.5.hmm -o /home/tc229954/subset/training_data0.5/emission/training_data/
