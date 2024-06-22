import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm
from numpy import identity as eye
import matplotlib.pyplot as plt
import random
from Calculation import Calculation

# 使用時はcal = Calculationのインスタンス必要


class KF:
    def __init__(self, calculation, tuning=1.10, obs_point=40, method=0):
        self.F = 8.0
        self.N = 40
        self.dt = 0.05
        self.days = 365
        self.day_steps = int(0.20 / self.dt)
        self.time_step = self.days * self.day_steps
        self.ls_time_step = [i for i in range(self.time_step)]
        self.IN = np.eye(self.N)
        self.obs_point = obs_point
        self.df_x_true = pd.read_csv('t_data.csv', header=0, index_col=0)
        self.df_y = pd.read_csv('o_data.csv', header=0, index_col=0)
        self.H = np.eye(self.N)
        self.R = np.eye(self.N)
        self.cal = calculation
        self.method = method
        self.delta = 0.001
        self.tuning = tuning
        self.a = 4.959494059253832
        self.Pa = (self.a ** 2) * np.eye(self.N)

    def run_simulation(self):
        x_true = self.df_x_true.values
        y = self.df_y.values
        x_a = x_true[1, :]
        forecast, analysis = [], []

        y, self.H, self.R = self.cal.obs_remove(
            obs_point=self.obs_point, y=y, H=self.H, R=self.R, method=1)

        for i in range(self.time_step):
            x_f = self.cal.Rk4(x_a)
            forecast.append(x_f)
            JM = np.zeros((self.N, self.N))
            for j in range(self.N):
                JM[:, j] = (self.cal.Rk4(x_a + self.delta *
                            self.IN[:, j]) - self.cal.Rk4(x_a)) / self.delta
            Pf = self.tuning * JM @ self.Pa @ JM.T
            K = Pf @ self.H.T @ np.linalg.pinv(self.H @ Pf @ self.H.T + self.R)
            x_a = x_f + K @ (y[i, :] - self.H @ x_f)
            analysis.append(x_a)
            self.Pa = (np.eye(self.N) - K @ self.H) @ Pf

        return self.results(forecast, analysis, x_true, y)

    def results(self, forecast, analysis, x_true, y):
        forecast = np.array(forecast)
        analysis = np.array(analysis)
        error_f, error_a = [], []

        for i in range(self.time_step):
            xf = forecast[i, :]
            xa = analysis[i, :]
            xt = x_true[i, :]
            epsilon_f = xf - xt
            epsilon_a = xa - xt
            error_f.append(self.cal.RMS(epsilon_f))
            error_a.append(self.cal.RMS(epsilon_a))

        return error_a, error_f
