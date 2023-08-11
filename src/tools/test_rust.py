import my_rust_module
import numpy as np
from random import choice
from string import ascii_lowercase, digits
import pdb
from collections import defaultdict
import h5py

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
            print(f"idx {match_idx}")
            print(f"indices {indices_array[match_idx]}")
            #print(f"score {match_scores}")
            names = unrolled_names[
                indices_array[match_idx]
            ]  # the names of the targets for each 1000 hits
            print(f"names: {names}")
            #print(f"scores: {match_scores}")
            sorted_match_idx = np.argsort(match_scores)[::-1]
            
            
            _, unique_indices = np.unique(names[sorted_match_idx], return_index=True)
            new_indices = list(
                indices_array[match_idx][sorted_match_idx][unique_indices]
            )
            print(f"scores: {match_scores[sorted_match_idx]}")
            new_scores = list(match_scores[sorted_match_idx][unique_indices])
            #print(f"names: {names[sorted_match_idx]}")
            #print(f"scores: {match_scores[sorted_match_idx]}")
            print(f"indices: {unique_indices}")
            
            for distance, name in zip(new_scores, unrolled_names[new_indices]):
                filtered_scores[name] += distance
        filtered_scores_list.append(filtered_scores)

    return filtered_scores_list


#scores_array = np.random.random(size=(3, 100))
#indices_array = np.random.randint(0, 3, size=(3, 100))

scores_array = np.load("testscores.npy")
indices_array = np.load("testindices.npy")
with h5py.File("/xdisk/twheeler/daphnedemekas/all_scores.h5", "w") as hf:
    for i, arr in enumerate([scores_array]):
        hf.create_dataset(f"array_{i}", data=arr)

    with h5py.File("/xdisk/twheeler/daphnedemekas/all_indices.h5", "w") as hf:
        for i, arr in enumerate([indices_array]):
            hf.create_dataset(f"array_{i}", data=arr)


# scores_array = np.random.random(size=(5, 100, 1000))
# indices_array = np.random.randint(0, 100, size=(5, 100, 1000))

chars = ascii_lowercase + digits
#unrolled_names = np.array(
#    ["".join(choice(chars) for _ in range(2)) for _ in range(100)]
#)

unrolled_names = np.array(['z0', 'ci', 'ry', 'qp', 'vx', 'zn', '3j', 'gf', 'kq', 'nf', 'qr',
       '45', 'kr', 'jp', 'ty', '1o', 'mc', 'ue', 'ee', '32', 'fv', '5f',
       '31', 'l1', 'ts', 'iv', 'fx', 'qt', 'of', '47', '0t', 'so', 'it',
       'wi', 'mg', 'fr', 'qw', '7d', 'mf', 'ct', 'm8', 'to', 'cr', 'zq',
       '7q', 'kc', 'rl', 'fz', '6y', 'rg', 'kv', 'z2', 'dj', 'jv', 'r5',
       'su', 'fj', 'r3', 'xt', 'tu', '19', 'fo', 'sh', '2z', 'qi', 'a7',
       'll', 'aq', 'qj', 'h0', 't0', 'q2', 'qf', 'wz', 'cb', 'x0', '2h',
       'i4', 'ut', '9t', 'nu', 'pf', 'gq', 'r0', '8q', 'ab', '90', 'ya',
       'lk', 'si', 'ta', 'ex', '1e', 'r8', '03', 'lp', '84', '5f', 'k2',
       'bm'], dtype='<U2')

print(scores_array)
scores_list = my_rust_module.filter_scores(
    [scores_array], [indices_array], unrolled_names
)
# print(scores_list)

pdb.set_trace()
scores_list_og = filter_scores([scores_array], [indices_array], unrolled_names)
#pdb.set_trace()
