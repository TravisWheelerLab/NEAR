#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/may13/transformer_embedding_layer/default/version_0/"
model_name="epoch_3_0.066049.ckpt"
echo $model_root $model_name

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
--compute_accuracy --n_seq_per_target_family 3 \
 --embed_dim 1280  \
--min_seq_len 256 --index_device "cuda:1"


