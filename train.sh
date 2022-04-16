#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=emission
#SBATCH --mem=500000
#SBATCH --gres=gpu:4
#SBATCH --output=contrastive.out
#SBATCH --error=contrastive.err
#SBATCH --exclude=compute-1-1


cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

time $py_cmd -m prefilter train\
    --gpus 3 \
    --num_nodes 1 \
    --num_workers 0 \
    --log_dir models/contrastive/exps_apr15/per_batch_loss_with_negs \
    --data_path /home/tc229954/data/prefilter/pfam/seed/clustered/0.5 \
    --logo_path /home/tc229954/data/prefilter/pfam/seed/clustered/0.5/\
    --batch_size 12 \
    --epochs 200 \
    --learning_rate 1e-3 \
    --check_val_every_n_epoch 1 \
    --max_pool \
    --real_data \
