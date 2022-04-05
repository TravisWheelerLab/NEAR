#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"


model_root="models/contrastive/exps_mar31/with_emission/default/version_1/"
model_name="epoch_223_5.348292.ckpt"
figure_path="eval_with_emission.png"

time $py_cmd prefilter/utils/evaluation_utils.py "$model_root" "$model_name" "$figure_path" -i -t
