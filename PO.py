import pandas as pd
import numpy as np
from scipy.linalg import inv, sqrtm
from numpy import identity as eye
import matplotlib.pyplot as plt
import random
from Calculation import Calculation

# 使用時はcal = Calculationのインスタンス必要
cal = Calculation()


class PO:
    def __init__(self, calculation=cal, obs_point=40, method=0):
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
        # 以下新変数
        self.member = 200

    def run_simulation(self):
        x_true = self.df_x_true.values
        y = self.df_y.values
        ua = np.zeros((self.N, self.member))
        uf = np.zeros((self.N, self.member))
        dxf = np.zeros((self.N, self.member))
        error_f, error_a = [], []
        # アンサンブルメンバー作成
        for m in range(self.member):
            # それぞれ初期に乱数擾乱加えてモデル回す
            ua[:, m] = np.random.rand(self.N) + self.F
            for i in range(self.time_step):
                ua[:, m] = self.cal.Rk4(ua[:, m])

        for i in range(self.time_step):
            # forecast step
            for m in range(self.member):
                uf[:, m] = self.cal.Rk4(ua[:, m])
            dxf = uf - np.mean(uf, axis=1, keepdims=True)
            Pf = (dxf @ dxf.T) / (self.member - 1)

            # analysis step
            K = Pf @ self.H.T @ np.linalg.inv(self.H @ Pf @ self.H.T + self.R)
            for m in range(self.member):
                ua[:, m] = uf[:, m] + \
                    K @ ((y[i, :] + np.random.rand(self.N)) -
                         (self.H @ uf[:, m]))
            error_a.append(np.linalg.norm(
                x_true[i, :] - np.mean(ua, axis=1)) / np.sqrt(self.N))
            error_f.append(np.linalg.norm(
                x_true[i, :] - np.mean(uf, axis=1)) / np.sqrt(self.N))

        return error_a, error_f
