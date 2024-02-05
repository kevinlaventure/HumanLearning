import numpy as np
from numba import jit
from typing import Tuple


class MonteCarlo:

    def __init__(self, num_path: int, num_simulation: int, num_path_per_year: int = 252):
        """
        Initialize a MonteCarlo object
        @param num_path: number of path
        @param num_simulation: number of simulation
        @param num_path_per_year: number of path in a year, defaults to 252.
        """
        self.num_path = num_path
        self.num_simulation = num_simulation
        self.num_path_per_year = num_path_per_year
        # np.random.seed(seed=7)

    def univariate_gbm(self, st: float, iv: float, d: float, b: float, r: float):
        """
        Generate univariate geometric brownian motion
        @param st: stock price
        @param iv: implied volatility
        @param d: continuously compounded dividend yield
        @param b: continuously compounded repo rate or borrowing cost
        @param r: continuously compounded risk-free interest rate
        @return: single asset generated timeseries
        """
        dt = 1 / self.num_path_per_year

        ts = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st, dtype=float)

        z = np.random.standard_normal(size=(self.num_path, self.num_simulation))

        drift = r - d - b
        for i in range(1, self.num_path + 1):
            ts[i] = ts[i - 1] * np.exp((drift - 0.5 * iv ** 2) * dt + iv * np.sqrt(dt) * z[i - 1])

        return ts

    @staticmethod
    @jit(nopython=True)
    def _calculate_bivariate_path(num_path, ts_1, ts_2, drift_1, drift_2, iv1, iv2, dt, z):
        for i in range(1, num_path + 1):
            ts_1[i] = ts_1[i - 1] * np.exp((drift_1 - 0.5 * iv1 ** 2) * dt + iv1 * np.sqrt(dt) * z[i - 1, :, 0])
            ts_2[i] = ts_2[i - 1] * np.exp((drift_2 - 0.5 * iv2 ** 2) * dt + iv2 * np.sqrt(dt) * z[i - 1, :, 1])
        return ts_1, ts_2

    def bivariate_gbm(self,
                      st1: float, iv1: float, q1: float, b1: float,
                      st2: float, iv2: float, q2: float, b2: float,
                      rho: float, r: float) -> Tuple[np.array, np.array]:
        """
        Generate bivariate geometric brownian motion
        @param st1: asset 1 stock price
        @param iv1: asset 1 implied volatility
        @param q1: asset 1 continuously compounded dividend yield
        @param b1: asset 1 continuously compounded repo rate or borrowing cost
        @param st2: asset 2 stock price
        @param iv2: asset 2 implied volatility
        @param q2: asset 2 continuously compounded dividend yield
        @param b2: asset 2 continuously compounded repo rate or borrowing cost
        @param rho: implied correlation between asset 1 and asset 2
        @param r: continuously compounded risk-free interest rate
        @return: 2 correlated assets generated timeseries
        """
        dt = 1 / self.num_path_per_year
        mu = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])

        ts_1 = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st1, dtype=float)
        ts_2 = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st2, dtype=float)

        z = np.random.multivariate_normal(mean=mu, cov=cov, size=(self.num_path, self.num_simulation))

        drift_1 = r - q1 - b1
        drift_2 = r - q2 - b2

        ts_1, ts_2 = self._calculate_bivariate_path(self.num_path, ts_1, ts_2, drift_1, drift_2, iv1, iv2, dt, z)

        return ts_1, ts_2
