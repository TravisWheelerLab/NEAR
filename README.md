# Paper
See [https://www.biorxiv.org/content/10.1101/2024.01.25.577287](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i449/8199346)
Instructions and scripts needed to train NEAR and recreate the results from the NEAR paper can be found at https://github.com/TravisWheelerLab/NEAR/blob/2025_paper_evaluation

# Prefilter
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Finding good alignment candidates for homology search.

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
