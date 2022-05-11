#!/usr/bin/env bash

py_cmd="time /home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/contrastive/may10/mlm_task/default/version_0/"
model_name="epoch_40_0.419092.ckpt"
echo $model_root $model_name
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --visualize --clustered_split --save_self_examples --n_images 5
# echo "visualizing"
# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
#  --visualize --clustered_split --n_seq_per_target_family 1\
#  --pretrained_transformer --n_neighbors 50 --n_images 100 --image_path debug --save_self_examples

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
--compute_accuracy --clustered_split --n_seq_per_target_family 1\
 --n_neighbors 20


