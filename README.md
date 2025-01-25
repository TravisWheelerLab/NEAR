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
conda install numpy matplotlib Bio -c pytorch -c nvidia faiss-gpu=1.9.0 anaconda::pyyaml
conda install conda-forge::maturin
maturin release
conda deactivate
```

## Running NEAR

```bash
conda activate near_env
python3 src/near_embed.py 
```

`python3 src/evaluate.py {configfile}` where configfile is the name of the config file that you changed. 
