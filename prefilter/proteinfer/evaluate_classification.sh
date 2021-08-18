#!/bin/bash

root='/home/tom/Dropbox/multilabel_classification_logs/lightning_logs/'
python evaluate_model.py\
    --logs_dir $root/version_12\
    --gpus 1
