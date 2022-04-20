#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"

model_root="models/contrastive/exps_apr19/with_real_masked_sequences/default/version_0"
model_name="epoch_0_5.685787.ckpt"
figure_path="test.png"

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 1.0 --indel_rate 0.1
