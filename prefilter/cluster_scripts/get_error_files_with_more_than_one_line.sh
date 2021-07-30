#!/bin/bash
for i in hmmer-output/*err; 
do 
    a=$(wc -l $i)
    b=$(echo $a | awk -F' ' '{print $1}')
    if [[ $b != 0 ]]; then
	echo $i;
    fi
done
