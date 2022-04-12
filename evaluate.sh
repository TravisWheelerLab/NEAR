#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"


model_root="models/contrastive/all_vs_all/exps_apr8/non_diag_alignment/default/version_2/"
model_name="epoch_7_81.051033.ckpt"
figure_path="test.png"

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" --embed_dim 256
