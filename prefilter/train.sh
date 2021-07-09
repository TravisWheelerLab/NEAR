#! /bin/bash

python w2v_train.py\
    --log_dir /home/tom/Dropbox/lightning_logs\
    --model_name model-max-seq-length-256.pt\
    --batch_size 10\
    --epochs 10000\
    --res_block_n_filters 64\
    --res_block_kernel_size 3\
    --vocab_size 23\
    --n_res_blocks 5\
    --res_bottleneck_factor 0.5\
    --embedding_dim 128\
    --max_sequence_length 5\
    --data_path /home/tom/share/prefilter/data/small-dataset\
    --normalize_output_embedding\
    --initial_learning_rate 1e-3\
    --num_workers 10\
    --gamma 0.9\
    --n_negative_samples 5\
    --gpus 1\
    --pooling_layer_type max\
    --check_val_every_n_epoch 500\
    --auto_lr_find
