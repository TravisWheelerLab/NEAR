# !/usr/bin/env bash

set -e
pid=0.5

for f in training_data/$pid/*train.fa
do 
    x=$(grep ">" $f | grep " PF" | wc -l)
    y=$(grep ">" $relabeled | grep " RLPF" | wc -l)
    if [[ $y != $x ]]
    then
	echo $y $x
    fi
done
