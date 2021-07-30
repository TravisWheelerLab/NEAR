#!/bin/bash

#SBATCH --partition=wheeler_lab_small_cpu
#SBATCH --job-name=json
#SBATCH --output=json-output/json-%a.out
#SBATCH --error=json-output/json-%a.err
#SBATCH --array=[1-18102]%100

f=$(sed -n "$SLURM_ARRAY_TASK_ID"p array_job_files.txt)

module load python3
bash convert_domtblout_to_json.sh $f Pfam-A.hmm 0.65
