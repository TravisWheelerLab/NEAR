#!/bin/bash

FASTA_FILE=$1
DOMTBLOUT=$2
OUT_DIR=$3

# Takes sequences from a fasta file, grabs their corresponding
# records in the domtblout, and save the sequenes and their labels 
# to json files in out directory.

if [[ $# != 3 ]]; then
    echo -n "usage: convert_domtblout_to_json.sh "
    echo "<fasta file> <domtblout file> <out_path>"
    exit
fi

extension="${FASTA_FILE##*.}"
base=$(basename $FASTA_FILE)
filename="${base%.*}"

JSON=$OUT_DIR/$filename.json
echo $JSON $DOMTBLOUT $FASTA_FILE

python3 create_json_labels_from_hmmer_output.py\
	--domtblout $DOMTBLOUT\
	--sequences $FASTA_FILE\
	--label-fname $JSON\
	--overwrite\
	--evalue-threshold 1e-5
