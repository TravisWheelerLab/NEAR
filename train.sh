#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=emission
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --output=contrastive.out
#SBATCH --error=contrastive.err


cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
#    --decoy_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/shuffled_training_data/200_file_subset\

time $py_cmd -m prefilter train\
    --gpus 4 \
    --num_nodes 1 \
    --num_workers 32 \
    --log_dir models/contrastive/exps_mar23 \
    --data_path /home/tc229954/max_hmmsearch/200_file_subset \
    --logo_path /home/tc229954/data/prefilter/pfam/seed/clustered/0.5/\
    --batch_size 64 \
    --epochs 10000 \
    --learning_rate 1e-3 \

