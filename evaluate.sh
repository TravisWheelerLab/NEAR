#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

model_root="models/contrastive/exps_apr27/better_sub_dists/no_max_pool_no_indels/default/version_0/"
model_name="epoch_99_1.309933.ckpt"

echo "no indels no max pool"
time $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --visualize --n_images 1000


