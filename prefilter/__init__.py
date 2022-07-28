"""
Prefilter passes good candidates to hmmer.
"""
import pdb
from types import SimpleNamespace
import os
from argparse import ArgumentParser
from prefilter.config import ex

__version__ = "0.0.1"

MASK_FLAG = -1
DROP_FLAG = -2
DECOY_FLAG = "DECOY"

array_job_template = """#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --output=array_out.out
#SBATCH --error=ERR
#SBATCH --nodes=1
#SBATCH --array=[1-ARRAY_JOBS]%200
DEPENDENCY
#SBATCH --cpus-per-task=1
#SBATCH --exclude=compute-1-11

f=$(sed -n "$SLURM_ARRAY_TASK_ID"p ARRAY_INPUT_FILE)
echo $f
RUN_CMD
"""
single_job_template = """#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_small_cpu,wheeler_lab_large_cpu
#SBATCH --output=single_job_out.out
#SBATCH --nodes=1
DEPENDENCY
#SBATCH --cpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=compute-1-11

RUN_CMD
"""

@ex.automain
def main(_config):
    from prefilter.train import main
    args = SimpleNamespace(**_config)
    pdb.set_trace()
    main(args)

