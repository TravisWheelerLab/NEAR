#!/bin/bash
# This script will run the test/train splitting from an MSA (or set of MSAs)
set -e # quit on error
hmm_database=/home/tc229954/pfam_label_creation/pfam/Pfam-A.hmm # where are the HMMs
# stored?
msa_file=subset-msa.msa # Where's the MSA used to create the HMMs?
sequence_file=subset-seqs.fa # the file containing the sequences you 
# used to create the MSA (no repeats allowed)
test_sequence_database=one-seq.fa # this can literally be a file of garbage with
# one long sequence. 
# esl-sfetch --index $sequence_file
profmark_out_dir=profmark_out
profmark_name=pmark
hmmer_out=hmmer
hmmer_train="$hmmer_out"-train
hmmer_test="$hmmer_out"-test
hmmer_val="$hmmer_out"-val
data_directory=dataset

p="$profmark_out_dir/""$profmark_name"
mkdir -p $profmark_out_dir

# esl-sfetch --index $sequence_file
# esl-sfetch --index $test_sequence_database
# create-profmark -N 0 -1 0.5 -2 0.5 -F 0.01 $p $msa_file $test_sequence_database
# # if you get an error that says  
# # ''can't construct test seq; no db seq >= 815 residues''
# # open your garbage sequence database and add a few lines.
# esl-reformat fasta $p.msa | grep ">" > train_names.txt
# grep ">" $p.fa > test_names.txt
# sed 's/>//g' train_names.txt > train_names_raw.txt
# sed 's/>//g' test_names.txt > test_names_raw.txt
# # 
# sed 's/.*domains: //g' test_names_raw.txt > test_names_stripped.txt
# sed 's/ /\n/g' test_names_stripped.txt > test_names_raw.txt
# 
# read words l c <<< $(wc test_names_raw.txt);
# 
# esl-selectn $((words / 2)) test_names_raw.txt > test_names_.txt
# 
# grep -Fvx -f test_names_.txt test_names_raw.txt > validation_names.txt
# 
# mv test_names_.txt  test_names_raw.txt
# 
# 
# esl-sfetch -f $sequence_file train_names_raw.txt > train_sequences.txt
# esl-sfetch -f $sequence_file test_names_raw.txt > test_sequences.txt
# esl-sfetch -f $sequence_file validation_names.txt > validation_sequences.txt
#  
# sed -E 's/ PF.*//g' $sequence_file > $sequence_file.reformat
#  
# hmmsearch -E 100.0 -o /dev/null --noali --domtblout $hmmer_train $hmm_database train_sequences.txt
# hmmsearch -E 100.0 -o /dev/null --noali --domtblout $hmmer_test $hmm_database test_sequences.txt
# hmmsearch -E 100.0 -o /dev/null --noali --domtblout $hmmer_val $hmm_database validation_sequences.txt

python3 seq_process.py --domtblout $hmmer_train --sequences train_sequences.txt --label-fname train-sequences-and-labels.json 
python3 seq_process.py --domtblout $hmmer_test --sequences test_sequences.txt --label-fname test-sequences-and-labels.json 
python3 seq_process.py --domtblout $hmmer_val --sequences validation_sequences.txt --label-fname val-sequences-and-labels.json 

mv *-labels.json $data_directory
