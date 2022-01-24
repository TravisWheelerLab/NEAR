#!/bin/bash


time python -m prefilter train\
    --gpus 4\
    --num_nodes 1\
    --num_workers 32\
    --log_dir with_emission_sequences_no_resampling \
    --emission_sequence_path "$HOME"/subset/training_data0.5/emission/training_data \
    --model_name model.pt\
    --data_path "$HOME"/subset/training_data0.5 \
    --decoy_path "$HOME"/tmp/ \
    --batch_size 32\
    --epochs 10000\
    --normalize_output_embedding\
    --learning_rate 1e-3\
    --check_val_every_n_epoch 10\
    --step_lr_step_size 5\
    --step_lr_decay_factor 0.1\
    --res_block_n_filters 1100\
    --vocab_size 23\
    --res_block_kernel_size 3\
    --res_bottleneck_factor 0.5\
    --n_res_blocks 5\
    --dilation_rate 2
