#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node 2
#SBATCH --nodes=2

# source ~/anaconda/bin/activate
# conda activate tf15

time python -m prefilter train\
    --gpus 1\
    --num_nodes 1\
    --num_workers 8\
    --log_dir "$HOME"/prefilter-logs/test \
    --model_name model.pt\
    --data_path "$HOME"/data/prefilter/training_data/0.35/100/ \
    --decoy_path "$HOME"/tmp/ \
    --batch_size 64\
    --epochs 1000\
    --single_label \
    --train_from_scratch\
    --layer_1_nodes 0\
    --layer_2_nodes 0\
    --normalize_output_embedding\
    --learning_rate 1e-3\
    --check_val_every_n_epoch 1\
    --pos_weight 1\
    --step_lr_step_size 5\
    --step_lr_decay_factor 0.1\
    --res_block_n_filters 1100\
    --vocab_size 23\
    --res_block_kernel_size 7\
    --res_bottleneck_factor 0.5\
    --n_res_blocks 5\
    --dilation_rate 2\
    --resample_families \
    --resample_based_on_uniform_dist
