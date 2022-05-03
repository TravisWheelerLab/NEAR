#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=with_mlp
#SBATCH --gres=gpu:4
#SBATCH --output=mlp.out

cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

time $py_cmd -m prefilter train\
    --gpus 3 \
    --num_nodes 1 \
    --num_workers 0 \
    --log_dir models/contrastive/exps_may2/mlp_batch_norm_large_model \
    --uniprot_file /home/tc229954/data/prefilter/uniprot/uniprot_sprot.fasta \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --check_val_every_n_epoch 1 \
    --real_data \
    --apply_mlp



