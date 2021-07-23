#!/bin/bash

# first, check if hmmfile exists:
FAMILY=$1
PFAM_A=$2
HMM_DIRECTORY=$3
FASTA_DIRECTORY=$4

if [[ ! -f $HMM_DIRECTORY/$FAMILY.hmm ]]; then
    hmmfetch -o $HMM_DIRECTORY/$FAMILY.hmm $PFAM_A $FAMILY
else
    echo "$HMM_DIRECTORY/$FAMILY.hmm already created :)"
fi


# Options controlling what to emit:
#   -a : emit alignment
#   -c : emit simple majority-rule consensus sequence
#   -C : emit fancier consensus sequence (req's --minl, --minu)
#   -p : sample sequences from profile, not core model

hmmemit -c -o $FASTA_DIRECTORY/$FAMILY.consensus.fa $HMM_DIRECTORY/$FAMILY.hmm
