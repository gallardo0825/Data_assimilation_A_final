import numpy as np
ua = np.zeros((40, 200))
uf = np.zeros((40, 200))
dxf = np.zeros((40, 200))
for m in range(200):
    # それぞれ初期に乱数擾乱加えてモデル回す
    ua[:, m] = np.random.rand(40) + 8
print(np.random.rand(40))
