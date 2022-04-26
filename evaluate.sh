#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

model_root="models/contrastive/exps_apr22/debugging/default/version_7/"
model_name="epoch_37_4.159367.ckpt"

time $py_cmd prefilter/evaluate.py "$model_root" "$model_name" --compute_accuracy
