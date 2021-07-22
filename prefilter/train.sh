#! /bin/bash

python w2v_train.py --log_dir /home/tom/Dropbox/lightning_logs\
    --gpus 0\
    --batch_size 10\
    --epochs 1\
    --res_block_n_filters 150\
    --res_block_kernel_size 3\
    --vocab_size 23\
    --n_res_blocks 10\
    --res_bottleneck_factor 0.5\
    --embedding_dim 1024\
    --data_path /home/tom/share/prefilter/data/small-dataset\
    --normalize_output_embedding\
    --initial_learning_rate 1e-3\
    --num_workers 10\
    --gamma 0.9\
    --n_negative_samples 2\
    --device cpu\
    --pooling_layer_type max\
    --check_val_every_n_epoch 10\
    --model_name model.pt
