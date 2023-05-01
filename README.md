# prefilter
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Passing good candidates to the forward/backward algorithm using NNs.

In particular, we aim to emulate HMMER.


## Data pipeline 

The training and evaluation data comes from UniRef90, where we randomly sampled 1,000,000 sequences. 
We partitioned these sequences such that 100,000 of them are labelled 'query' sequences and 900,000 of them are labelled 'target' sequences. 

We ran HMMER on all query-to-target pairs and stored the stdout files containing the alignments, as well as the e-values.

### Train-Test split 
To construct the training and evaluation datasets, we needed to make sure that there were no sequences appearing in training and evaluation data that had a high percent similarity, as this is likely to lead to overfitting. To ensure this, we used the UCLUST package (https://drive5.com/usearch/manual/uclust_algo.html) to cluster our dataset, and separated the clustered data such that entire clusters appear together and no sequence from the same cluster are distributed across training and evaluation data. 

### Training data
The training data comes from HMMER stdout files - we use the alignments made by HMMER as input sequence pairs into our model such that aminos are aligned with corresponding aminos. 

### Multi-positives 
We also include the option to include multiple positives which allows one to use the Supervised Constrastive Loss. This changes the training data such that we find all overlapping sections within all the query alignment subsequences, and match the corresponding target subsequences such that they are used as positives in the SCL. 

## Running the training pipeline 

1) First collect your data from UniRef or any other source. You can separate out your data as you like, but make sure all target and query fasta files are in separate directories. The paths to these directories should be put in the `config.yaml` file. 
2) Install the UCLUST package and run clustering on your target data: 

`python3 -m src.data.tools.cluster_target_data --target_fasta {} --cluster_savedir {} --train_savepath {} --eval_savepath {}`

3) Run `phmmer` on your data, and save this to a stdoutfile: 

`python3 -m src.data.tools.run_hmmer --query_fasta {} --target_fasta {} --tbloutfile {} --stdoutfile {}`

4) Now parse the stdout file to construct the training dataset 





4) 
