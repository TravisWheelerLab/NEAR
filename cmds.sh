#!/usr/bin/env bash

set -e

eval_cmd="python prefilter/evaluate.py"
hparams=with_emission_sequences_no_resampling/default/version_2/hparams.yaml
model=with_emission_sequences_no_resampling/default/version_2/checkpoints/epoch=29-val/loss=0.00167-val/f1=0.67264.ckpt
summ_cmd="python prefilter/data_scripts/summarize_dataset.py"

# summarize the dataset
# $eval_cmd "recall" $model $hparams  validation_recall.png
# $eval_cmd "f1score" $model $hparams recall_neigh.png -c 0.05
# $eval_cmd "recall_per_family" $model $hparams recall_per_family_all.png
$eval_cmd "recall_per_family" $model $hparams recall_per_family_just_neighborhood.png --just_neighborhood

# $summ_cmd "neigh" $hparams neighborhood_labels_val.png --key val_files
# $summ_cmd "neigh" $hparams neighborhood_labels_train.png --key train_files

# $summ_cmd "comp" $hparams comparison.png
# $summ_cmd "uniq" $hparams num_seq_per_uniq_label_combination_neighborhood.png

