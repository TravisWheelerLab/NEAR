import h5py
import time
import numpy as np
import tqdm
import pdb
def reduce_indices(indices, names, index_mapping):
    # indices is of shape (seq len, 1000)
    new_indices = np.zeros_like(indices)

    for i, amino_index_list in tqdm.tqdm(enumerate(indices)):
        
        #new_indices[i] = np.array(
        #    [names.index(unrolled_names[idx]) for idx in amino_index_list]
        #)
        new_indices[i] = np.array([index_mapping[idx] for idx in amino_index_list])

    return new_indices



print("load targets")

with open("/xdisk/twheeler/daphnedemekas/prefilter/target_names.txt", "r") as f:
    target_names = f.readlines()
    target_names = [t.strip("\n") for t in target_names]

with open("/xdisk/twheeler/daphnedemekas/prefilter/target_lengths.txt", "r") as f:
    target_lengths = f.readlines()
    target_lengths = [int(t.strip("\n")) for t in target_lengths]

#unrolled_names = np.repeat(target_names, target_lengths)
#print(len(unrolled_names))

print(len(target_lengths))
print(len(target_names))
index_mapping = {}

target_idx = 0
j = 0
for length in target_lengths:
    for i in range(length):
        k = i + j
        index_mapping[k] = target_idx
    j += length
    target_idx += 1   

print(len(index_mapping))
print(33009630)

print("reducing...")
reduced_indices = []
with h5py.File("/xdisk/twheeler/daphnedemekas/all-indices.h5", "r") as f:
    # Assuming you know the dataset name in your .h5 file is 'dataset_name'
    i = 0
    print(i)
    while f"array_{i}" in f:
        myarray = f[f"array_{i}"][:]
        reduced_array = reduce_indices(myarray, target_names, index_mapping)
        reduced_indices.append(reduced_array)
        i += 1
print("Saving new indices")
with h5py.File("/xdisk/twheeler/daphnedemekas/new-indices.h5", "w") as hf:
    for i, arr in enumerate(new_indices):
        hf.create_dataset(f"array_{i}", data=arr)

