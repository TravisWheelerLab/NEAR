#!/bin/bash

HMM_DATABASE=$1
SEQUENCES=$2
DOMTBLOUT=$3
JSON_OUT=$4

if [ $# != 4 ]; then
    echo "usage: ./get-labels-from-hmmer.sh <HMM_DATABASE> <SEQUENCE_DATABASE> <DOMTBLOUT_FNAME> <JSON_OUT_FNAME>"
    exit
fi

filename=$(basename -- "$SEQUENCES")
extension="${filename##*.}"
filename="${filename%.*}"
FASTA_FILENAME=$filename.fa
STOCKHOLM_FILENAME=$filename.sto

if [[ 'afa' == $extension ]]; then
    if [[ ! -f $filename.fa ]]; then
	esl-reformat -o $STOCKHOLM_FILENAME stockholm $SEQUENCES
	esl-reformat -o $FASTA_FILENAME fasta $STOCKHOLM_FILENAME
    fi
fi

if [[ ! -f $DOMTBLOUT ]]; then 
    hmmsearch -E 100.0\
	-o /dev/null\
	--noali \
	--domtblout $DOMTBLOUT\
	$HMM_DATABASE\
	$filename.fa
    else
	echo "$DOMTBLOUT already created, not running hmmsearch"
fi

python3 create_json_labels_from_hmmer_output.py\
    --domtblout $DOMTBLOUT\
    --sequences $FASTA_FILENAME\
    --label-fname $JSON_OUT
