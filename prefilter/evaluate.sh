#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=evaluate_model
#SBATCH --output=evaluated
#SBATCH --gres=gpu:1

source ~/anaconda/bin/activate
conda activate tf15

LOG_ROOT=/home/tc229954/model-from-scratch/small-medium-dataset/single-best/lightning_logs/

python evaluate_ranking_model.py --save_prefix /home/tc229954/gpu_8 \
       --logs_dir $LOG_ROOT/version_2917104 \
       --model_path $LOG_ROOT/version_2917104/checkpoints/epoch=27-val_loss=0.98813.ckpt \
       --batch_size 128
