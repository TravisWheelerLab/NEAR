#!/bin/bash

python model.py\
    --log_dir /home/tom/Dropbox/w2v_train_dl_embeddings/\
    --model_name model.pt\
    --data_path /home/tom/share/prefilter/data/small-dataset/json\
    --batch_size 8\
    --epochs 10000\
    --embedding_dim 512\
    --layer_1_nodes 1024\
    --layer_2_nodes 512\
    --normalize_output_embedding\
    --initial_learning_rate 1e-3\
    --num_workers 10\
    --n_negative_samples 5\
    --gpus 1\
    --check_val_every_n_epoch 500\
