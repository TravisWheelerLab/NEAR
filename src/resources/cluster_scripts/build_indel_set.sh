#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=hmmbuild-%a
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=outfiles/hmmbuild-%a.out
### REQUIRED. Specify the PI group for this job (twheeler).
#SBATCH --account=twheeler
### the arizona cluster system: standard (uses group's monthly allocation of
### resources), windfall (does NOT use up your monthly quota, but jobs run in
### this partition can be interrupted), high_priority (requires purchasing
### compute resources), and qualified. You'll probably want to use one of
### <standard,windfall>
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=12
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=5gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=12:01:00
### any other slurm options are supported, but not required
#SBATCH --array=[1-23]%23


cd /home/u4/colligan/share/prefilter/src/resources
source $HOME/miniconda3/bin/activate
conda activate faiss
export LD_LIBRARY_PATH=$HOME/miniconda3/lib/:$LD_LIBRARY_PATH
# final query_seq len of 16
name=$(sed -n "$SLURM_ARRAY_TASK_ID"p split_fasta_names.txt)
echo "running with name: $name"
python build_indel_set.py "$name"