#!/bin/bash

#SBATCH --partition=wheeler_lab_large_cpu,wheeler_lab_small_cpu
#SBATCH --job-name=hmmsearch
#SBATCH --output=output/hmmsearch-%a.out
#SBATCH --error=output/hmmsearch-%a.err
#SBATCH --array=[1-18102]%100

f=$(sed -n "$SLURM_ARRAY_TASK_ID"p array_job_files.txt)
bash run-hmmer.sh $f Pfam-A.hmm
