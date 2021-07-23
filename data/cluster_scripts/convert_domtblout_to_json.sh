#!/bin/bash

FASTA_FILE=$1
DOMTBLOUT=$2
OUT_DIR=$3

if [[ $# != 3 ]]; then
    echo "usage: convert_domtblout_to_json.sh <.afa file>\ 
    <percent-id-threshold-to-analyze>"
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
	--label-fname $JSON
