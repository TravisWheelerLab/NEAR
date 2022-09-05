#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=benchmark
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=benchmark-faiss-%x-%j.out
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
#SBATCH --ntasks=32
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=5gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=05:01:00
### any other slurm options are supported, but not required.

cd /home/u4/colligan/share/prefilter/
source "$HOME"/miniconda3/bin/activate
conda activate faiss

root="/home/u4/colligan/data/prefilter/uniref_benchmark/"

echo "1k15k"
time evaluate with hit_filename="$root""/1k15k_hits2.txt" query_file="$root""/Q_benchmark1k30k.fa" target_file="$root""/T_benchmark2k15k.fa"

echo "2k30k"
time evaluate with hit_filename="$root""/2k30k_hits2.txt" query_file="$root""/Q_benchmark2k30k.fa" target_file="$root""/T_benchmark2k30k.fa"

echo "1k30k"
time evaluate with hit_filename="$root""/1k30k_hits2.txt" query_file="$root""/Q_benchmark1k30k.fa" target_file="$root""/T_benchmark2k30k.fa"

echo "2k15k"
time evaluate with hit_filename="$root""/2k15k_hits2.txt" query_file="$root""/Q_benchmark2k30k.fa" target_file="$root""/T_benchmark2k15k.fa"

