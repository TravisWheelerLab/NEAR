import os
import json

from glob import glob
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

train = glob('./small-dataset/json/*train*')
test = glob('./small-dataset/json/*test*')

for f in train:

    of = os.path.splitext(os.path.basename(f))[0]
    of = os.path.join('./small-dataset/train_subset/', of+'.fa')
    with open(f, 'r') as src:
        sequences_and_labels = json.load(src)

    records = []
    i = 0
    j = 0
    for sequence, labels in sequences_and_labels.items():
        for label in labels:
            records.append(SeqRecord(Seq(sequence.upper()), label,
                description=str(i)))
            i += 1
        if j > 9:
            break
        j += 1
    SeqIO.write(records, of, 'fasta')
