#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
# model_root="models/contrastive/exps_apr27/better_sub_dists/max_pool_indels/default/version_0/"
# model_name="epoch_90_4.149399.ckpt"
# echo "max pool and indels"
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split
#
# model_root="models/contrastive/exps_apr27/better_sub_dists/max_pool_no_indels/default/version_0/"
# model_name="epoch_91_0.960423.ckpt"
# echo "no indels with max pool"
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split
#
# model_root="models/contrastive/exps_apr27/better_sub_dists/no_max_pool_indels/default/version_0/"
# model_name="epoch_91_5.559702.ckpt"
# echo "indels, no max pool"
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split

model_root="models/contrastive/exps_apr27/better_sub_dists/no_max_pool_no_indels/default/version_2/"
model_name="epoch_15_1.210274.ckpt"
echo "no indels no max pool"
$py_cmd prefilter/evaluate.py "$model_root" \
            "$model_name" --compute_accuracy --n_neighbors 50 --clustered_split --n_seq_per_target_family 50

# model_root="models/contrastive/exps_apr28/conv_pool_no_indels/default/version_0/"
# model_name="epoch_99_0.777703.ckpt"
# echo "no indels with conv pool"
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split
#
# model_root="models/contrastive/exps_apr28/conv_pool_with_indels/default/version_0/"
# model_name="epoch_43_4.277881.ckpt"
# echo "indels with conv pool"
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy --clustered_split
