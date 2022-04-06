#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"


model_root="models/all_vs_all/exps_apr5/default/version_2/"
model_name="epoch_41_-940.582825.ckpt"
figure_path="test.png"

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" --embed_dim 256
