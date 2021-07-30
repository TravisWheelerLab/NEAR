#!/bin/bash

python classification_model.py\
    --log_dir /home/tom/Dropbox/multilabel_classification_logs/\
    --model_name model-0.35.pt\
    --data_path /home/tom/share/prefilter/data/small-dataset/json/0.35\
    --batch_size 128\
    --epochs 30\
    --layer_1_nodes 1024\
    --layer_2_nodes 512\
    --normalize_output_embedding\
    --initial_learning_rate 1e-3\
    --num_workers 10\
    --gpus 1\
    --check_val_every_n_epoch 2\
    --pos_weight 1\
