import json
from glob import glob

files = glob('1k/*json')
files += glob('10k/*json')
files += glob('100k/*json')
files += glob('1m/*json')

set_of_sets = set()

for f in files:
    with open(f, 'r') as src:
        dct = json.load(src)

    for sequence, label_set in dct.items():
        if isinstance(label_set, list):
            set_of_sets.add(frozenset(label_set))


print(len(set_of_sets))
