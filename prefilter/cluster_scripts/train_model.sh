#!/bin/bash

#SBATCH --partition=wheeler_lab_gpu
#SBATCH --job-name=prot2vec
#SBATCH --output=why.out
#SBATCH --gres=gpu:2

source $HOME/anaconda/bin/activate/
conda activate pt
python $HOME/share/prefilter/prefilter/w2v_train.py --gpus 2 --log_dir /home/tc229954/ --batch_size 32
