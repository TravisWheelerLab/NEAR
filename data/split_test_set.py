import json
import sys
import os
import numpy as np

from glob import glob

if __name__ == '__main__':

    dirs = sys.argv[1]

    test_files = glob(os.path.join(dirs, '*test.json'))

    for f in test_files:

        valid_filename = f.replace('test.json', 'valid-split.json')
        test_filename = f.replace('test.json', 'test-split.json')
        with open(f, 'r') as src:
            sequence_to_label = json.load(src)

        sequences = list(sequence_to_label.keys())

        if len(sequences) > 1:
            valid = np.random.choice(sequences,
                                     size=int(len(sequences)*0.5), 
                                     replace=False)
            
            valid_sequence_to_label = {}
            for v in valid:
                valid_sequence_to_label[v] = sequence_to_label[v]
                del sequence_to_label[v]
            with open(valid_filename, 'w') as dst:
                json.dump(valid_sequence_to_label, dst)

            with open(test_filename, 'w') as dst:
                json.dump(sequence_to_label, dst)


        else:
            print('only 1 seq in test, not splitting into valid (but saving nonetheless!)')
            with open(test_filename, 'w') as dst:
                json.dump(sequence_to_label, dst)
