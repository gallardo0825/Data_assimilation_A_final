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
enkf = EnKF()
enkf_a, enkf_f = enkf.run_simulation()
ls = range(len(enkf_a))
plt.figure(figsize=(10, 6))
plt.xlabel('Time step')
plt.ylabel('RMS of absolute value of epsilon')
plt.plot(ls, enkf_a, label="analysis")
plt.plot(ls, enkf_f, label="forecast")
plt.legend()
plt.show()

# %%
po = PO()
po_a, po_f = po.run_simulation()
ls = range(len(po_a))
plt.figure(figsize=(10, 6))
plt.xlabel('Time step')
plt.ylabel('RMS of absolute value of epsilon')
plt.plot(ls, po_a, label="analysis")
plt.plot(ls, po_f, label="forecast")
plt.legend()
plt.show()
