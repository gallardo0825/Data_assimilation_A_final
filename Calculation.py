import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm
from numpy import identity as eye
import matplotlib.pyplot as plt
import random


class Calculation:
    def __init__(self, F=8.0, N=40, dt=0.05):
        self.F = F
        self.N = N
        self.dt = dt
        self.u = np.full(self.N, self.F) + np.random.rand(self.N)

    def L96(self, x):
        N = self.N
        F = self.F
        dxdt = np.zeros((N))
        for i in range(2, N-1):
            dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        dxdt[0] = (x[1] - x[N-2]) * x[N-1] - x[0] + F
        dxdt[1] = (x[2] - x[N-1]) * x[0] - x[1] + F
        dxdt[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1] + F
        return dxdt

    def Rk4(self, xold):
        dt = self.dt
        k1 = self.L96(xold)
        k2 = self.L96(xold + k1 * dt / 2.)
        k3 = self.L96(xold + k2 * dt / 2.)
        k4 = self.L96(xold + k3 * dt)
        xnew = xold + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return xnew

    def RMS(self, x):
        return np.sqrt(np.mean(x**2))

    def trRMS(self, x):
        return np.sqrt(x / self.N)

    def column_means(self, array):
        return np.mean(array, axis=0).tolist()

    def obs_remove(self, obs_point, y, H, R, method=0):
        rmv_point = self.N - obs_point
        methods = [0, 1, 2, 3]
        if method in methods:
            if method == 0:
                return y, H, R
            # 連続抜き
            elif method == 1:
                indices_to_remove = np.s_[0:rmv_point]
                y = np.delete(y, indices_to_remove, axis=1)
                H = np.delete(H, indices_to_remove, axis=0)
                R = np.delete(R, indices_to_remove, axis=0)
                R = np.delete(R, indices_to_remove, axis=1)
                return y, H, R

            # 等間隔抜き
            elif method == 2:
                indices_to_remove = np.linspace(
                    0, self.N - 1, rmv_point, dtype=int)
                y = np.delete(y, indices_to_remove, axis=1)
                H = np.delete(H, indices_to_remove, axis=0)
                R = np.delete(R, indices_to_remove, axis=0)
                R = np.delete(R, indices_to_remove, axis=1)
                return y, H, R

            else:
                indices_to_remove = np.random.choice(
                    self.N, rmv_point, replace=False)
                y = np.delete(y, indices_to_remove)
                H = np.delete(H, indices_to_remove, axis=0)
                R = np.delete(R, indices_to_remove, axis=0)
                R = np.delete(R, indices_to_remove, axis=1)
                return y, H, R

        else:
            print("Wrong method was selected")
