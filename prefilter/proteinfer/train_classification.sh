#!/bin/bash

python classification_model.py\
    --log_dir $HOME/multilabel_classification_logs/\
    --model_name model-0.35.pt\
    --data_path $HOME/data/prefilter/small-dataset/related_families/json/0.5\
    --batch_size 1024\
    --epochs 30\
    --layer_1_nodes 1024\
    --layer_2_nodes 512\
    --normalize_output_embedding\
    --initial_learning_rate 1e-3\
    --num_workers 32\
    --gpus 1\
    --check_val_every_n_epoch 2\
    --pos_weight 1\
