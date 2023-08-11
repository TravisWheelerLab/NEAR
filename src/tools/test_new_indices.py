import h5py
#from src.evaluators.contrastive_functional import filter_scores
import pdb 
import numpy as np
from collections import defaultdict
def filter_scores(
    scores_array_list: np.array, indices_array_list: np.array, unrolled_names: np.array
) -> dict:
    """Filters the scores such that every query amino can only
    be matched to one amino from each target sequence
    and it matches the one with the biggest score.

    Then sums the scores that belong to the same target
    and returns the resulting distances in a dict

    scores_array (numqueryaminos, 1000): an array of 1000 scores per query amino
    indices_array: (numqueryaminos, 1000) the indices of the target sequence name (in unrolled_names)
    for each of the scores in scores_array
    unrolled_names: an array of target names that the indices in indices_array correspond to
    """

    filtered_scores_list = []
    for scores_array, indices_array in zip(scores_array_list, indices_array_list):
        filtered_scores: dict = defaultdict(float)

        # iterate over query amino scores
        for match_idx in range(len(scores_array)):
            match_scores = scores_array[match_idx]
 #           print(f"idx {match_idx}")
 #           print(f"indices {indices_array[match_idx]}")
            #print(f"score {match_scores}")
            names = unrolled_names[
                indices_array[match_idx]
            ]  # the names of the targets for each 1000 hits
  #          print(f"names: {names}")
            #print(f"scores: {match_scores}")
            sorted_match_idx = np.argsort(match_scores)[::-1]
            
            
            _, unique_indices = np.unique(names[sorted_match_idx], return_index=True)
            new_indices = list(
                indices_array[match_idx][sorted_match_idx][unique_indices]
            )
   #         print(f"scores: {match_scores[sorted_match_idx]}")
            new_scores = list(match_scores[sorted_match_idx][unique_indices])
            #print(f"names: {names[sorted_match_idx]}")
            #print(f"scores: {match_scores[sorted_match_idx]}")
    #        print(f"indices: {unique_indices}")
            
            for distance, name in zip(new_scores, unrolled_names[new_indices]):
                filtered_scores[name] += distance
        filtered_scores_list.append(filtered_scores)

    return filtered_scores_list


old_indices = []
with h5py.File("/xdisk/twheeler/daphnedemekas/all-indices-test-FULL.h5", "r") as f:
    # Assuming you know the dataset name in your .h5 file is 'dataset_name'
    i = 0
    while f"array_{i}" in f:
        old_indices.append(f[f"array_{i}"][:])
        i += 1


indices = []
with h5py.File("/xdisk/twheeler/daphnedemekas/all-indices-test-REDUCED.h5", "r") as f:
    # Assuming you know the dataset name in your .h5 file is 'dataset_name'
    i = 0
    while f"array_{i}" in f:
        indices.append(f[f"array_{i}"][:])
        i += 1

print("load targets")

with open("/xdisk/twheeler/daphnedemekas/prefilter/target_names.txt", "r") as f:
    target_names = f.readlines()
    target_names = [t.strip("\n") for t in target_names]

print("load unrolled names")
with open("/xdisk/twheeler/daphnedemekas/unrolled_names.txt", "r") as f:
    unrolled_names = [line.strip("\n") for line in f.readlines()]


print("load scores")
scores = []
with h5py.File("/xdisk/twheeler/daphnedemekas/all-scores-test.h5", "r") as f:
    # Assuming you know the dataset name in your .h5 file is 'dataset_name'
    i = 0
    while f"array_{i}" in f:
        scores.append(f[f"array_{i}"][:])
        i += 1

unrolled_names = np.array(unrolled_names)
target_names = np.array(target_names)
scores_list_og = filter_scores(scores, old_indices, unrolled_names)

new_scores_list = filter_scores(scores, indices, target_names)


assert scores_list_og == new_scores_list
