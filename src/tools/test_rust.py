import my_rust_module
import numpy as np
from random import choice
from string import ascii_lowercase, digits
import pdb
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
            names = unrolled_names[
                indices_array[match_idx]
            ]  # the names of the targets for each 1000 hits
            sorted_match_idx = np.argsort(match_scores)[::-1]

            _, unique_indices = np.unique(names[sorted_match_idx], return_index=True)
            new_indices = list(
                indices_array[match_idx][sorted_match_idx][unique_indices]
            )
            new_scores = list(match_scores[sorted_match_idx][unique_indices])

            for distance, name in zip(new_scores, unrolled_names[new_indices]):
                filtered_scores[name] += distance
            filtered_scores_list.append(filtered_scores)

    return filtered_scores_list


scores_array = np.random.random(size=(100, 1000))
indices_array = np.random.randint(0, 100, size=(100, 1000))


# scores_array = np.random.random(size=(5, 100, 1000))
# indices_array = np.random.randint(0, 100, size=(5, 100, 1000))

chars = ascii_lowercase + digits
unrolled_names = np.array(
    ["".join(choice(chars) for _ in range(2)) for _ in range(100)]
)

scores_list = my_rust_module.filter_scores(
    [scores_array], [indices_array], unrolled_names
)
# print(scores_list)


scores_list_og = filter_scores([scores_array], [indices_array], unrolled_names)
pdb.set_trace()
