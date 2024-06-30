# %%
import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm
from numpy import identity as eye
import matplotlib.pyplot as plt
import random
from Calculation import Calculation
from KF import KF
from EnKF import EnKF
from PO import PO

# %%
po = PO()
po_1, po_2 = po.run_simulation()
ls = range(len(po_1))
plt.figure(figsize=(10, 6))
plt.xlabel('Time step')
plt.ylabel('RMS of absolute value of epsilon')
plt.plot(ls, po_2, label="PO")
plt.legend()
plt.show()
# %%
