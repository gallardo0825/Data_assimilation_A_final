import numpy as np
from calculation import Calculation

cal = Calculation()

i_indices = np.arange(5)[:, None]  # [[0], [1], [2]]
j_indices = np.arange(5)           # [0, 1, 2]

localization_mat = np.vectorize(cal.localization)(i_indices, j_indices)
print(localization_mat)
