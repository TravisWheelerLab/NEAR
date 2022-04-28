#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=no_mpool
#SBATCH --gres=gpu:1
#SBATCH --output=max_pool.out

cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

time $py_cmd -m prefilter train\
    --gpus 1 \
    --num_nodes 1 \
    --num_workers 0 \
    --log_dir models/contrastive/exps_apr28/conv_pool_with_indels \
    --uniprot_file /home/tc229954/data/prefilter/uniprot/uniprot_sprot.fasta \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --check_val_every_n_epoch 1 \
    --real_data \
    --max_pool \
    --apply_indels


