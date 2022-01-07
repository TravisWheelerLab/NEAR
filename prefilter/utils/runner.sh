#! /usr/bin/env bash

#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --output=array_out.out
#SBATCH --nodes=1
#SBATCH --array=[1-35734]%200
#SBATCH --cpus-per-task=1


f=$(sed -n "$SLURM_ARRAY_TASK_ID"p names.txt)
echo $f

/home/tc229954/anaconda/envs/prefilter/bin/python \
    /home/tc229954/share/prefilter/prefilter/utils/label_fasta.py label \
    $f \
    clustered/Pfam-A.0.5-train.hmm \
    -o training_data/0.5-train/

