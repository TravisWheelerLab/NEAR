#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=gpu-benchmark
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=gpu-benchmark-faiss-%x-%j.out
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
#SBATCH --mem-per-cpu=4gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=05:01:00
### any other slurm options are supported, but not required.
set -e

cd /home/u4/colligan/share/prefilter/
source "$HOME"/miniconda3/bin/activate
conda activate faiss
export LD_LIBRARY_PATH=$HOME/miniconda3/lib/:$LD_LIBRARY_PATH

root="/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/"

for filter_value in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  evaluate with hit_filename="/xdisk/twheeler/colligan/data/prefilter/uniref_benchmark/2k30k_hits_IVF_gpu20pct.txt" filter_value="$filter_value"
done

