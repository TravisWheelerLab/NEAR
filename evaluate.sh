#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/contrastive/exps_may3/alignment_aware/default/version_11"
model_name="epoch_11_0.909499.ckpt"
echo "post-mlp, no max pool, no indels, long train"

#$py_cmd prefilter/evaluate.py "$model_root" "$model_name" --visualize --image_path 20pid --clustered_split
$py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split --n_seq_per_target_family 9
