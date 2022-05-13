#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=with_mlp
#SBATCH --gres=gpu:4
#SBATCH --output=mlp.out

cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

time $py_cmd -m prefilter train\
    --gpus 4 \
    --num_nodes 1 \
    --num_workers 0 \
    --afa_path /home/tc229954/data/prefilter/panthr/afa/ \
    --log_dir models/may12/distill_30k_esm_embeddings \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 1e-3 \
    --apply_mlp \
    --distill





