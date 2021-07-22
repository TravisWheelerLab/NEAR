#!/bin/bash

ali=$1
HMM_DATABASE=$2
PID=$3

if [[ $# != 3 ]]; then
    echo "usage: process_labels.sh <.afa file> <hmm-database>\
    <percent-id-threshold-to-analyze>"
fi

extension="${ali##*.}"
filename="${ali%.*}"

FASTA_FILENAME=$filename.fa

JSON_TEST=$filename-$PID-test.json
JSON_TRAIN=$filename-$PID-train.json

DOMTBLOUT=$filename.domtblout
FASTA_TEST=$filename.$PID"-test".fa

FASTA_TRAIN=$filename.$PID"-train".fa

if [ -e $FASTA_TRAIN ]; then

python3 $HOME/prefilter/prefilter/utils/create_json_labels_from_hmmer_output.py\
    --domtblout $DOMTBLOUT\
    --sequences $FASTA_TRAIN\
    --label-fname $JSON_TRAIN
fi

if [ -e $FASTA_TEST ]; then
    python3 $HOME/prefilter/prefilter/utils/create_json_labels_from_hmmer_output.py\
	--domtblout $DOMTBLOUT\
	--sequences $FASTA_TEST\
	--label-fname $JSON_TEST
fi

echo $DOMTBLOUT $FASTA_TEST $JSON_TEST
