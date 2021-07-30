#!/bin/bash

#SBATCH --partition=wheeler_lab_large_cpu,wheeler_lab_small_cpu
#SBATCH --job-name=hmmemit
#SBATCH --output=hmmemit-outputs/hmmemit-%a.out
#SBATCH --error=hmmemit-outputs/hmmemit-%a.err
#SBATCH --array=[1-18260]%100

f=$(sed -n "$SLURM_ARRAY_TASK_ID"p pfam_hmms/pfam_names.txt)
echo $f
bash get_consensus_sequences.sh $f pfam_hmms/Pfam-A.hmm pfam_hmms/hmm pfam_hmms/consensus_sequences
