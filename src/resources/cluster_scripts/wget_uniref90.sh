#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=wget-%a
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=wget-%a.out
### REQUIRED. Specify the PI group for this job (twheeler).
#SBATCH --account=twheeler
### the arizona cluster system: standard (uses group's monthly allocation of
### resources), windfall (does NOT use up your monthly quota, but jobs run in
### this partition can be interrupted), high_priority (requires purchasing
### compute resources), and qualified. You'll probably want to use one of
### <standard,windfall>
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=1
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=5gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=24:01:00
### any other slurm options are supported, but not required

mkdir /xdisk/twheeler/colligan/uniref
cd /xdisk/twheeler/colligan/uniref
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz

