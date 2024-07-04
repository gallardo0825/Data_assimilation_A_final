import numpy as np


def localization(i, j, dist=2.0):
    return np.exp(-(i - j) ** 2 / dist)


i_indices = np.arange(40)[:, None]  # [[0], [1], [2]]
j_indices = np.arange(40)           # [0, 1, 2]

localization_mat = np.vectorize(localization)(i_indices, j_indices)
print(localization_mat)
