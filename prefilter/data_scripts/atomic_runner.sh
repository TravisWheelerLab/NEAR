#!/bin/bash

AFA_DIRECTORY=$1
PID=$2
OUT_PATH_FASTA=$3
OUT_PATH_JSON=$4
DOMTBLOUT_DIRECTORY=$5

if [[ $# != 5 ]]; then
    echo "not enough arguments"
    exit
fi

for afa in $AFA_DIRECTORY/*afa;
do
    extension="${afa##*.}"
    base=$(basename $afa)
    filename="${base%.*}"

    if [[ ! -e $AFA_DIRECTORY/$filename.ddgm ]]; then
	echo "couldn't find $AFA_DIRECTORY/$filename.ddgm, run carbs cluster"
	continue
    else
	echo -n ""
	# carbs split -T argument --split_test --output_path $OUT_PATH_FASTA $afa $PID
    fi

    train_file_success=$OUT_PATH_FASTA/$filename.$PID-train.fa;
    train_file_failure=$OUT_PATH_FASTA/$filename.-train.fa;
    test_file=$OUT_PATH_FASTA/$filename.$PID-test.fa;
    valid_file=$OUT_PATH_FASTA/$filename.$PID-valid.fa;

    echo $filename
    
    if [[ -e $train_file_success ]]; then
	# if the train file was successfully split, assume that test was too
	# check for valid b/c there's no valid split if there's only one
	# sequence in test
	bash convert_domtblout_to_json.sh $train_file_success $DOMTBLOUT_DIRECTORY/$filename.domtblout $OUT_PATH_JSON
	bash convert_domtblout_to_json.sh $test_file $DOMTBLOUT_DIRECTORY/$filename.domtblout $OUT_PATH_JSON
	if [[ -e $valid_file ]]; then
	bash convert_domtblout_to_json.sh $valid_file $DOMTBLOUT_DIRECTORY/$filename.domtblout $OUT_PATH_JSON
	fi
    else 
	bash convert_domtblout_to_json.sh $train_file_failure $DOMTBLOUT_DIRECTORY/$filename.domtblout $OUT_PATH_JSON
    fi
done
