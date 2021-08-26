#!/bin/bash
#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=train
#SBATCH --output=more_workers.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=32


PID=0.5
source ~/anaconda/bin/activate
conda activate tf15
python classification_model.py\
    --log_dir $HOME/multilabel_classification_logs/\
    --model_name model-0.35.pt\
    --data_path $HOME/data/prefilter/small-dataset/related_families/json/$PID\
    --batch_size 32\
    --epochs 300\
    --layer_1_nodes 1024\
    --layer_2_nodes 512\
    --normalize_output_embedding\
    --initial_learning_rate 1e-3\
    --num_workers 32\
    --gpus 1\
    --check_val_every_n_epoch 2\
    --pos_weight 1\
    --num_workers 0