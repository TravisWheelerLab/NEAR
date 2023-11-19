import pickle
import os

all_hits_max_file_4 = "data/hmmerhits-masked"
with open(all_hits_max_file_4 + ".pkl", "rb") as file:
    all_hits_max_4 = pickle.load(file)

filtered_target_names = "data/filtered_target_names.txt"

target_names_file = "target_names.txt"

all_targets = set()

for query in all_hits_max_4:
    targets = list(all_hits_max_4[query].keys())
    all_targets.update(targets)

with open(target_names_file, "r") as file:
    target_names = file.readlines()

print(f"First 5 target names : {target_names[:5]}")

filtered_targets = []
for target in target_names:
    target = target.strip()

    if target not in all_targets:
        filtered_targets.append(target)

print(f"Found {len(filtered_targets)} filtered targets ")
with open(filtered_target_names, "w") as file:
    for target in filtered_targets:
        file.write(target + "\n")
