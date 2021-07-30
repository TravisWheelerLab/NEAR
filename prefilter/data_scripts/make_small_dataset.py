import json
import os
import shutil
import numpy as np

from glob import glob
from random import shuffle

test_files = sorted(glob('./1k/*0.2-test-split.json'))
train_files_original = sorted(glob('./1k/*train.json'))
# 1502866 total train sequences in 1k/
# 34765 total sequences in 1k/

test_ = []
tot =0
tot2 = 0
shuffle(train_files_original)
train_files = train_files_original[:int(len(train_files_original)*0.01)]

for train in train_files:
    base = train.replace('-train.json', '')
    with open(train, 'r') as src:
        tot += len(json.load(src))
    for test in test_files:
        if base in test:
            test_.append(test)
            break


for test in test_:
    with open(test, 'r') as src:
        tot2 += len(json.load(src))

print(tot2/tot, tot2, tot)

for train in train_files:
    of = os.path.basename(train)
    of = os.path.join('./small-dataset', of)
    shutil.copyfile(train, of)

for test in test_:
    of = os.path.basename(test)
    of = os.path.join('./small-dataset', of)
    shutil.copyfile(test, of)
