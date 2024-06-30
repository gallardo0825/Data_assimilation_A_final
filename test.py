import numpy as np

uf = np.arange(1, 26).reshape(5, 5)
dXf = uf - np.mean(uf, axis=1, keepdims=True)
print(dXf)
