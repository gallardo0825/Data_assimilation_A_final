import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm
from numpy import identity as eye
import matplotlib.pyplot as plt
import random
from Calculation import Calculation
from KF import KF

kf = KF()
kf_a, kf_f = kf.run_simulation()
ls = range(len(kf_a))
plt.figure(figsize=(10, 6))
plt.xlabel('Time step')
plt.ylabel('RMS of absolute value of epsilon')
plt.plot(ls, kf_a, label="KF")
plt.show()
