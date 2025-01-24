# Paper
See https://www.biorxiv.org/content/10.1101/2024.01.25.577287v1.full.pdf

# Prefilter
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Finding good alignment candidates for homology search.

## Disclaimer

This project is a working prototype. It is not (yet) a polished tool meant for general purpose use. 


## Requirements and setup

If your system has a working installation of Python3, Conda, and Rust, you can install NEAR's setup NEAR like so:

```bash
conda create --name near_env 
conda activate near_env
conda install numpy matplotlib Bio -c pytorch -c nvidia faiss-gpu=1.9.0
conda install conda-forge::maturin
maturin release
```

## Evaluating
To run on custom query and target sequences, edit one of the config/ files with your specifications. 
Namely, edit the the `query_file` and  `target_file` parameters to specify your data. 
The target embedding information will be saved off to the specified `target_embeddings`, `target_names` and `target_lengths` paths to speed up future searches. 

To speed up the search, if you have multiple CPUs, you can set `num_threads` to the nucmber of CPUs, which will parallelize embedding creation and search. 

Increasing `nprobe` will lead to more accurate results, but will be slower. 

You can run evaluation with 

`python3 src/evaluate.py {configfile}` where configfile is the name of the config file that you changed. 
