#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=with_mlp
#SBATCH --gres=gpu:4
#SBATCH --output=mlp.out

cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

time $py_cmd -m prefilter train\
    --gpus 1 \
    --num_nodes 1 \
    --num_workers 0 \
    --log_dir models/may20/circular_padding/ \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 1e-4 \
    --msa_transformer \
    --seq_len 128
