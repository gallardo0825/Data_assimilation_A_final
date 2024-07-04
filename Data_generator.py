import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm
from numpy import identity as eye
import matplotlib.pyplot as plt
import random
from Calculation import Calculation


class Datagenerater:
    def __init__(self, calculation):
        self.cal = calculation
        self.N = calculation.N
        self.u = np.full(self.N, self.cal.F) + np.random.rand(self.N)

    def Data_generate(self, Time_Step=1460):
        t_data, o_data = [], []
        u = self.u

        # Warm-up phase to eliminate transients
        for _ in range(Time_Step):
            u = self.cal.Rk4(u)

        # Data generation phase
        for _ in range(Time_Step):
            u = self.cal.Rk4(u)
            t_data.append(u)
            noisy_observation = u + np.random.randn(self.N)
            o_data.append(noisy_observation)

        # Save data to CSV
        pd.DataFrame(t_data).to_csv('t_data.csv')
        pd.DataFrame(o_data).to_csv('o_data.csv')
