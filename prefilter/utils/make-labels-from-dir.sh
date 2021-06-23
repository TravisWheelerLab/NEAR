#!/bin/bash

DIRECTORY=$1
HMM_DATABASE=$2
PID=$3

if [[ $# != 3 ]]; then
    echo "usage: make-labels-from-dir.sh <directory-to-analyze> <hmm-database>\
    <percent-id-threshold-to-analyze>"

for ali in $DIRECTORY/*afa;

do 
    extension="${ali##*.}"
    filename="${ali%.*}"

    FASTA_FILENAME=$filename.fa
    STOCKHOLM_FILENAME=$filename.sto

    JSON_TEST=$filename-test.json
    JSON_TRAIN=$filename-train.json
    DOMTBLOUT_TEST=$filename-test.domtblout
    DOMTBLOUT_TRAIN=$filename-train.domtblout
    FASTA_TEST=$filename.$PID"-test".fa
    FASTA_TRAIN=$filename.$PID"-train".fa

    if [ ! -e $FASTA_TRAIN ]; then
	echo $FASTA_TRAIN
	continue
    fi

    if [ ! -e $FASTA_TEST ]; then
	echo $FASTA_TEST
	continue
    fi

    if [[ ! -f $DOMTBLOUT_TRAIN ]]; then 
	hmmsearch -E 100.0\
	    -o /dev/null\
	    --noali\
	    --domtblout $DOMTBLOUT_TRAIN\
	    $HMM_DATABASE\
	    $FASTA_TRAIN
	else
	    echo "$DOMTBLOUT_TRAIN already created, not running hmmsearch"
    fi

    if [[ ! -f $DOMTBLOUT_TEST ]]; then 
	hmmsearch -E 100.0\
	    -o /dev/null\
	    --noali\
	    --domtblout $DOMTBLOUT_TEST\
	    $HMM_DATABASE\
	    $FASTA_TEST
	else
	    echo "$DOMTBLOUT_TEST already created, not running hmmsearch"
    fi


    python3 $HOME/prefilter/prefilter/utils/create_json_labels_from_hmmer_output.py\
	--domtblout $DOMTBLOUT_TEST\
	--sequences $FASTA_TEST\
	--label-fname $JSON_TEST

    python3 $HOME/prefilter/prefilter/utils/create_json_labels_from_hmmer_output.py\
	--domtblout $DOMTBLOUT_TRAIN\
	--sequences $FASTA_TRAIN\
	--label-fname $JSON_TRAIN

done
