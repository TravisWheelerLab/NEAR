#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=evaluate_transformer
#SBATCH --gres=gpu:1
#SBATCH --output=transformer_bsz_1.eval

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/may27/10_layer_resnet_no_batchnorm_large_sequences/default/version_0/"
model_name="epoch_17_2.024041.ckpt"
echo $model_root $model_name

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
--compute_accuracy --n_seq_per_target_family 1 \
--embed_dim 256  \
--seq_len -1 --batch_size 1 --index_device "cuda" --normalize_embeddings
