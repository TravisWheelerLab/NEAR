import sys
import numpy as np


if len(sys.argv) < 3:
    print("Usaged: mean_embedding.py [input files] output_file")
    exit(-1)

if len(sys.argv) == 3:
    in_files = [sys.argv[1]]

else:
    in_files = sys.argv[1:-1]

out_file = sys.argv[-1]

data = {}
for i, file_path in enumerate(in_files):
    print("Reading " + file_path)
    file_data = np.load(file_path)
    offset = len(data)
    for key in file_data:
        data[str(int(key) + offset)] = np.expand_dims(file_data[key].mean(axis=0), 0)

print("Saving to " + out_file)
np.savez(out_file, **data)