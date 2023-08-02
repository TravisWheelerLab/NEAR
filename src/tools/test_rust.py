import my_rust_module
import numpy as np
from random import choice
from string import ascii_lowercase, digits
from src.evaluators.contrastive_functional import filter_scores

scores_array = np.random.random(size=(100, 1000))
indices_array = np.random.randint(0, 100, size=(100, 1000))


# scores_array = np.random.random(size=(5, 100, 1000))
# indices_array = np.random.randint(0, 100, size=(5, 100, 1000))

chars = ascii_lowercase + digits
unrolled_names = ["".join(choice(chars) for _ in range(2)) for _ in range(100)]

scores_list = my_rust_module.filter_scores(
    [scores_array], [indices_array], unrolled_names
)
# print(scores_list)


scores_list_og = filter_scores()
