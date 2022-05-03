#!/usr/bin/env bash

set -e

# first need to run carbs with --save_ali or whatever the argument is
training_ali=$1
test_sequences=$2
# in .sto format
hmmbuild $training_ali.hmm $training_ali
# now search it.
hmmsearch --domtblout $training_ali.hmm test/$test_sequences.domtblout
