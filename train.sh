#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=model_from_scratch_compute-1-9
#SBATCH --output=model_from_scratch
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -w compute-1-9

source ~/anaconda/bin/activate
conda activate tf15

export NCCL_DEBUG=INFO

time python -m prefilter train\
    --gpus 1\
    --num_nodes 1\
    --num_workers 8\
    --log_dir "$HOME"/model-from-scratch/small-medium-dataset/single-best/\
    --model_name model.pt\
    --data_path "$HOME"/data/prefilter/small-medium-dataset/single-best/json/0.5\
    --batch_size 128\
    --epochs 30\
    --train_from_scratch\
    --layer_1_nodes 0\
    --layer_2_nodes 0\
    --normalize_output_embedding\
    --learning_rate 1e-2\
    --check_val_every_n_epoch 1\
    --pos_weight 1\
    --step_lr_step_size 5\
    --step_lr_decay_factor 0.1\
    --res_block_n_filters 550\
    --vocab_size 23\
    --res_block_kernel_size 7\
    --res_bottleneck_factor 0.5\
    --n_res_blocks 5\
    --dilation_rate 2\
    --resample_families
    #--tune_initial_lr
