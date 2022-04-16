#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"


model_root="models/contrastive/exps_apr15/per_example_loss_with_negs/default/version_1"
model_name="epoch_61_13.789524.ckpt"
figure_path="test.png"

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 1.0 --indel_rate 0.1
