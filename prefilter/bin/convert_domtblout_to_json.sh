#!/bin/bash

FASTA_FILE=$1
DOMTBLOUT=$2
OUT_DIR=$3
EVALUE_THRESHOLD=$4

# Takes sequences from a fasta file, grabs their corresponding
# records in the domtblout, and save the sequences and their labels
# to json files in out directory.

if [[ $# != 4 ]]; then
    echo -n "usage: convert_domtblout_to_json.sh "
    echo "<fasta file> <domtblout file> <out_path> <evalue_threshold>"
    exit
fi

extension="${FASTA_FILE##*.}"
base=$(basename $FASTA_FILE)
filename="${base%.*}"

JSON=$OUT_DIR/$filename.json

create_json_labels_from_hmmer_output.py\
	--domtblout $DOMTBLOUT\
	--sequences $FASTA_FILE\
	--label-fname $JSON\
	--overwrite\
	--evalue-threshold $EVALUE_THRESHOLD