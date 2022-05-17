#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/may13/transformer_embedding_layer/default/version_0/"
model_name="epoch_27_0.061847.ckpt"
echo $model_root $model_name

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
--compute_accuracy --n_seq_per_target_family 1 \
--embed_dim 1280  --pretrained_transformer \
--min_seq_len -1 --index_device "cpu" --quantize_index
