# prefilter
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Passing good candidates to the forward/backward algorithm using NNs.

In particular, we aim to emulate HMMER.


## Evaluating
To run on custom query and target sequences, edit one of the config/ files with your specifications. 
Namely, edit the the `query_file` and  `target_file` parameters to specify your data. 
The target embedding information will be saved off to the specified `target_embeddings`, `target_names` and `target_lengths` paths to speed up future searches. 

To speed up the search, if you have multiple CPUs, you can set `num_threads` to the number of CPUs, which will parallelize embedding creation and search. 

Increasing `nprobe` will lead to more accurate results, but will be slower. 

You can run evaluation with 

`python3 src/evaluate.py {configfile}` where configfile is the name of the config file that you changed. 
