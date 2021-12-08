#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=evaluate_model
#SBATCH --output=evaluated
#SBATCH --gres=gpu:1

source ~/anaconda/bin/activate
conda activate tf15

LOG_ROOT=/home/tc229954/testpre/lightning_logs/

python -m prefilter eval --save_prefix /home/tc229954/gpu_8 \
       --logs_dir $LOG_ROOT/version_2 \
       --model_path $LOG_ROOT/version_2/model.pt \
       --batch_size 128
