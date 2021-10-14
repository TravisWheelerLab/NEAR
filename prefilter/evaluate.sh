#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=evaluate_model
#SBATCH --output=evaluated
#SBATCH --gres=gpu:1

source ~/anaconda/bin/activate
conda activate tf15

LOG_ROOT=/home/tc229954/model-from-scratch/small-medium-dataset/single-best/lightning_logs/

python utils/evaluate_ranking_model.py --save_prefix /home/tc229954/gpu_8 \
       --logs_dir $LOG_ROOT/version_2916977 \
       --model_path $LOG_ROOT/version_2916977/checkpoints/epoch=19-val_loss=0.95010.ckpt \
       --batch_size 128
