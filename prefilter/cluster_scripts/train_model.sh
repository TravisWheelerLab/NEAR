#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=prot2vec
#SBATCH --output=overfit.out
#SBATCH --gres=gpu:2

echo $PWD
source $HOME/anaconda/bin/activate
conda activate pt
python $HOME/share/prefilter/prefilter/w2v_train.py --gpus 1 --log_dir /home/tc229954/ --batch_size 8 --epochs 1000
