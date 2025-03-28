# Paper
See https://www.biorxiv.org/content/10.1101/2024.01.25.577287

# This branch

This branch contains the code needed to train NEAR and recreate the experiments in the NEAR paper.

## Disclaimer

This project is a working prototype. It is not (yet) a polished tool meant for general purpose use. 


## Requirements and setup

If your system has a working installation of Python3, Conda, and Rust, you can install NEAR's setup NEAR like so:

```bash
git clone https://github.com/TravisWheelerLab/NEAR.git
cd NEAR
conda create --name near_env python=3.12
conda config --set channel_priority flexible 

conda activate near_env
                  

conda install pytorch
conda install numpy 

conda install conda-forge::matplotlib
conda install -c conda-forge biopython
conda install conda-forge::pyyaml


conda install maturin
maturin develop --release
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=11 numpy
conda deactivate
```

## Running NEAR



```bash
conda activate near_env

python src/near_embed.py resnet.yaml models/resnet_877_256.pt sparse_benchmark/sparse_query_softmask.fa queries.npz

python src/near_embed.py resnet.yaml models/resnet_877_256.pt sparse_benchmark/sparse_target_hardmask_mixed.fa targets.npz

python src/search.py -g -q queries.npz -t targets.npz -o hits.csv


conda deactivate near_env


```
The output "hits.csv" will have lines such as "1 15274 0.9138156"

Here, 1 is the first query sequence, 15274 is the target sequence, and 0.9138156 is the total score between 1 and 15274. The numbers are the 1-indexed order of sequences in the original fasta files. Softmasking can be done by also providing src/serarch.py with --query_sequence and --target_sequence (the fasta file you provide should be softmasked).
