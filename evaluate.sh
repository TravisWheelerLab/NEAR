#!/usr/bin/env bash

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"


model_root="models/contrastive/exps_apr14/max_pool/more_mutated/default/version_0/"
model_name="epoch_199_4.792615.ckpt"
figure_path="test.png"

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 1.0 --indel_rate 0.1
exit
time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 0.3 --indel_rate 0.3

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 0.5 --indel_rate 0.1

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 0.5 --indel_rate 0.3

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 0.5 --indel_rate 0.5

time $py_cmd prefilter/utils/ali_evaluation.py "$model_root" "$model_name" "$figure_path" \
        --embed_dim 256 --sub_rate 0.85 --indel_rate 0.05
