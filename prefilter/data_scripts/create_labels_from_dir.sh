# !/bin/bash

set -e

# Ugh another shell script to make labels.

AFA_DIRECTORY=$1
PID=$2
OUT_PATH_FASTA=$3
OUT_PATH_JSON=$4
DOMTBLOUT_DIRECTORY=$5

if [[ $# != 5 ]]; then
    echo "incorrect # of arguments"
    exit
fi

for f in $AFA_DIRECTORY/*afa;
do
    carbs split -T argument --split_test --output_path $OUT_PATH_FASTA $f $PID
done

for f in $OUT_PATH_FASTA/*fa;
do 
    x=$(basename $f)
    y=$(echo $x | awk -F. '{print $1}') 
    z=1k/$y.domtblout
    bash convert_domtblout_to_json.sh $f $z $OUT_PATH_JSON 
done
