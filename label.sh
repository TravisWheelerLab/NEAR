#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_large_cpu
#SBATCH --job-name=label
#SBATCH --nodes=1
#SBATCH --output=out/label-%A-%a.out
#SBATCH --error=out/label-%A-%a.err
#SBATCH --array=[1-10000]%50

names="/home/tc229954/data/prefilter/pfam/seed/training_data/emission/0.45_rel_ent/emission_names.txt"
inp=$(sed -n "$SLURM_ARRAY_TASK_ID"p $names)
pfam="/home/tc229954/data/prefilter/pfam/seed/clustered/Pfam-A.0.5-train.hmm"
for d in 0.45_rel_ent 0.55_rel_ent 0.65_rel_ent;
do
  echo $d
/home/tc229954/anaconda/envs/prefilter/bin/python prefilter/utils/label_fasta.py label /home/tc229954/data/prefilter/pfam/seed/training_data/emission/removed/$d/1000_file_subset/$inp $pfam \
    -o /home/tc229954/data/prefilter/pfam/seed/training_data/emission/labeled/$d/1000_file_subset/
done
