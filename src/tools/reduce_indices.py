import h5py
import time


def reduce_indices(indices, names, unrolled_names):
    new_indices = []
    for idx in indices:
        new_indices.append(names.index(unrolled_names[idx]))

    return new_indices


print("load indices")
# Open the file in 'read' mode

indices = []
with h5py.File("/xdisk/twheeler/daphnedemekas/all-indices.h5", "r") as f:
    # Assuming you know the dataset name in your .h5 file is 'dataset_name'
    i = 0
    while f"array_{i}" in f:
        indices.append(f[f"array_{i}"][:])
        i += 1


print(f"Len indices: {len(indices)}")

unrolled_names = []

print("load unrolled names")
with open("/xdisk/twheeler/daphnedemekas/unrolled-names.txt", "w") as f:
    unrolled_names = [line.strip("\n") for line in f.readliens()]

print("load targets")

with open("/xdisk/twheeler/daphnedemekas/prefilter/target_names.txt", "r") as f:
    target_names = f.readlines()
    target_names = [t.strip("\n") for t in target_names]

print("Reducing")
new_indices = []
for indices in new_indices:
    idx = reduce_indices(indices, target_names, unrolled_names)
    new_indices.append(idx)

print("Saving new indices")
with h5py.File("/xdisk/twheeler/daphnedemekas/new-indices.h5", "w") as hf:
    for i, arr in enumerate(new_indices):
        hf.create_dataset(f"array_{i}", data=arr)
