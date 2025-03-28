# Paper
See https://www.biorxiv.org/content/10.1101/2024.01.25.577287

## Disclaimer

This project is a working prototype. It is not (yet) a polished tool meant for general purpose use. 

## This branch

This branch contains the code needed to train NEAR and recreate the experiments in the NEAR paper.


## Requirements and setup

If your system has a working installation of Python3, Conda, and Rust, you can install NEAR's setup NEAR like so:

```bash
git clone --branch 2025_paper_evaluation https://github.com/TravisWheelerLab/NEAR.git
cd NEAR
conda env create -f near_eval_env.yml
```

## Running NEAR

```bash

python src/near_embed.py resnet.yaml models/resnet_877_256.pt sparse_benchmark/sparse_query_softmask.fa queries.npz
python src/near_embed.py resnet.yaml models/resnet_877_256.pt sparse_benchmark/sparse_target_softmask_mixed.fa targets.npz

python src/search.py -g -q queries.npz -t targets.npz --query_fasta sparse_benchmark/sparse_query_softmask.fa --target_fasta sparse_benchmark/sparse_target_softmask_mixed.fa -o hits.csv

```

The output "hits.csv" will have lines such as "1 15274 0.9138156"

Here, 1 is the first query sequence, 15274 is the target sequence, and 0.9138156 is the total score between 1 and 15274. The numbers are the 1-indexed order of sequences in the original fasta files. Softmasking can be done by also providing src/serarch.py with --query_sequence and --target_sequence (the fasta file you provide should be softmasked).
