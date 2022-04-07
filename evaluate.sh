#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"


model_root="models/all_vs_all/exps_apr5/supcon/default/version_5/"
model_name="epoch_8_1.942660.ckpt"
figure_path="test.png"

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" --embed_dim 256
