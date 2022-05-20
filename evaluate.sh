#!/usr/bin/env bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --nodes=1
#SBATCH --job-name=evaluate_transformer
#SBATCH --gres=gpu:1
#SBATCH --output=transformer_bsz_1.eval

py_cmd="/home/tc229954/anaconda/envs/prefilter/bin/python"
set -e
model_root="models/may20/circular_padding_and_single_msa_rep/default/version_0/"
model_name="epoch_3_3.064979.ckpt"
echo $model_root $model_name

$py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
--compute_accuracy --n_seq_per_target_family 1 \
--embed_dim 768 --plot_dots --msa_transformer \
--seq_len 128 --batch_size 16 --index_device "cuda" --normalize_embeddings

# $py_cmd prefilter/evaluate.py "$model_root" "$model_name" \
# --visualize --n_seq_per_target_family 1 \
# --embed_dim 256 --batch_size 1 --pretrained_transformer  \
# --min_seq_len -1 --index_device "cuda" --plot_dots --n_images 20
