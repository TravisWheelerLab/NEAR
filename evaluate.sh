#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/contrastive/exps_may2/mlp_batch_norm_large_model/default/version_0"
model_name="epoch_68_0.655258.ckpt"
echo "post-mlp, no max pool, no indels, long train"

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" --visualize --image_path 20pid --clustered_split
#$py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split --n_seq_per_target_family 9
