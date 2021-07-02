#!/bin/bash

#SBATCH --partition=wheeler_lab_large_cpu,wheeler_lab_small_cpu
#SBATCH --job-name=carbs
#SBATCH --output=carbs-output/carbs-%a.out
#SBATCH --error=carbs-output/carbs-%a.err
#SBATCH --array=[1-18102]%100

f=$(sed -n "$SLURM_ARRAY_TASK_ID"p array_job_files.txt)
carbs split -T argument $f 0.5 
