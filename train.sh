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
    --gpus 1 \
    --num_nodes 1 \
    --num_workers 0 \
    --log_dir /tmp/ \
    --data_path /home/tc229954/data/prefilter/pfam/seed/training_data/1000_file_subset/ \
    --logo_path /home/tc229954/data/prefilter/pfam/seed/clustered/0.5/\
    --batch_size 48 \
    --epochs 15 \
    --learning_rate 1e-3 \
    --check_val_every_n_epoch 2 \
    --supcon \
    --non_diag_alignment \
    --debug
