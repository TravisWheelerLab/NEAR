#!/usr/bin/env bash


f=clustered/0.5/CGI-121.0.5-train.fa

/home/tc229954/anaconda/envs/prefilter/bin/python \
    /home/tc229954/share/prefilter/prefilter/utils/label_fasta.py hdb $f Pfam-A.seed
