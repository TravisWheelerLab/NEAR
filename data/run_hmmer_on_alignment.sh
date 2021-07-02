#!/bin/bash

ali=$1
HMM_DATABASE=$2

set -e
extension="${ali##*.}"
filename="${ali%.*}"

FASTA_FILENAME=$filename.fa
STOCKHOLM_FILENAME=$filename.sto

JSON=$filename.json
DOMTBLOUT=$filename.domtblout
echo $DOMTBLOUT $FASTA_FILENAME $STOCKHOLM_FILENAME

if [[ ! -f $FASTA_FILENAME ]]; then
    esl-reformat stockholm $ali >> $STOCKHOLM_FILENAME
    esl-reformat fasta $STOCKHOLM_FILENAME >> $FASTA_FILENAME
fi

if [[ ! -f $DOMTBLOUT ]]; then 
    hmmsearch -E 100.0\
	-o /dev/null\
	--noali\
	--domtblout $DOMTBLOUT\
	$HMM_DATABASE\
	$FASTA_FILENAME
    else
	echo "$DOMTBLOUT already created, not running hmmsearch"
fi
