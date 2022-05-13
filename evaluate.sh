#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/may12/distill_30k_esm_embeddings/default/version_0/"
model_name="epoch_199_0.075835.ckpt"
echo $model_root $model_name

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
--compute_accuracy --n_seq_per_target_family 1 \
 --embed_dim 1280 \
--min_seq_len 256 --index_device cpu


