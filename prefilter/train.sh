#!/bin/bash
#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=model_from_scratch
#SBATCH --output=single_best_medium.out
#SBATCH --gres=gpu:1
#SBTACH --exclude="compute-1-3"
#SBATCH --nodes=1

# source ~/anaconda/bin/activate
# conda activate tf15
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONPATH=/home/tc229954/share/prefilter/
# nvidia-smi
# hostname

python train.py\
    --log_dir "$HOME"/model-from-scratch/small-medium-dataset/single-best/\
    --model_name model.pt\
    --data_path "$HOME"/data/prefilter/small-medium-dataset/single-best/json/0.5\
    --batch_size 32\
    --epochs 1000\
    --train_from_scratch\
    --layer_1_nodes 0\
    --layer_2_nodes 0\
    --normalize_output_embedding\
    --learning_rate 1e-3\
    --gpus 1\
    --check_val_every_n_epoch 1\
    --pos_weight 1\
    --num_workers 32\
    --resample_families\
    --step_lr_step_size 5\
    --step_lr_decay_factor 0.1\
    --res_block_n_filters 550\
    --vocab_size 23\
    --res_block_kernel_size 7\
    --res_bottleneck_factor 0.5\
    --n_res_blocks 5\
    --dilation_rate 2
    # --tune_initial_lr
    # --schedule_lr\
