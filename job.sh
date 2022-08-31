#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=brute_force
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=brute-force-hits-run-%x-%j.out
### REQUIRED. Specify the PI group for this job (twheeler).
#SBATCH --account=twheeler
### REQUIRED. Set the partition for your job. Four partitions are available in
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
#SBATCH --time=03:01:00
### any other slurm options are supported, but not required.
#SBATCH --gres=gpu:1

module load python/3.9
source ~/venvs/prefilter/bin/activate

cd /home/u4/colligan/share/prefilter/

evaluate with use_faiss=False hit_filename="without_faiss.txt"
