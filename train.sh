#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=emission
#SBATCH --gres=gpu:4
#SBATCH --output=emission.out

cd /home/tc229954/share/prefilter
py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
#    --decoy_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/shuffled_training_data/200_file_subset\

time $py_cmd -m prefilter train\
    --gpus 4 \
    --num_nodes 4 \
    --num_workers 32 \
    --log_dir models/exps_mar9 \
    --data_path /home/tc229954/max_hmmsearch/200_file_subset \
    --emission_sequence_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/0.45_rel_ent/200_file_subset \
    --emission_sequence_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/0.55_rel_ent/200_file_subset \
    --emission_sequence_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/0.65_rel_ent/200_file_subset \
    --emission_sequence_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/0.59_rel_ent/200_file_subset \
    --emission_sequence_path /home/tc229954/data/prefilter/pfam/seed/model_comparison/emission/0.5_rel_ent/200_file_subset \
    --model_name without_emission_sequences.pt \
    --batch_size 256 \
    --epochs 10000 \
    --learning_rate 1e-3 \
    --check_val_every_n_epoch 10 \
    --step_lr_step_size 5 \
    --step_lr_decay_factor 0.1 \
    --res_block_n_filters 256 \
    --vocab_size 23 \
    --res_block_kernel_size 3 \
    --res_bottleneck_factor 0.5 \
    --n_res_blocks 5 \
    --dilation_rate 2 \
    --subsample_neg_labels \
    --n_emission_sequences 50 \

