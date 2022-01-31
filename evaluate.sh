#!/usr/bin/env bash

set -e

hparams=with_emission_sequences_no_resampling/default/version_2/hparams.yaml
model=with_emission_sequences_no_resampling/default/version_2/checkpoints/epoch=15-val/loss=0.00140-val/f1=0.59126.ckpt
eval_cmd="python prefilter/evaluate.py"

# plot recall.
$eval_cmd "recall" $model $hparams primary_and_neighborhood_recall.png
echo "Done plotting recall."

# plot recall per family (all, neighborhood, and primary).
$eval_cmd "recall_per_family" $model $hparams recall_per_family.png
echo "Done plotting recall per family."
$eval_cmd "recall_per_family" $model $hparams recall_per_family_primary.png --just_primary
echo "Done plotting recall per family (primary labels only)."
$eval_cmd "recall_per_family" $model $hparams recall_per_family_neighborhood.png --just_neighborhood
echo "Done plotting recall per family (neighborhood labels only)."

$eval_cmd "recall_per_family" $model $hparams recall_per_family_neighborhood_with_emission.png --just_neighborhood \
 --emission_sequence_path /home/tc229954/subset/training_data0.5/emission/training_data/

# and finally the recall at each rank.
$eval_cmd "ranked_recall" $model $hparams ranked_recall.png
