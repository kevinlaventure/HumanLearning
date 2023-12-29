import numpy as np
from scipy.stats import norm
from typing import Union, Tuple
from scipy.integrate import quad
from abc import ABC, abstractmethod


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

        ts = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st)

        z = np.random.standard_normal(size=self.num_simulation)

        drift = r - d - b

        for i in range(1, self.num_path + 1):
            ts[i] = ts[i - 1] * np.exp((drift - 0.5 * iv ** 2) * dt + iv * np.sqrt(dt) * z[i - 1])

        return ts

    def bivariate_gbm(self,
                      st_1: float, iv_1: float, d_1: float, b_1: float,
                      st_2: float, iv_2: float, d_2: float, b_2: float,
                      rho: float, r: float) -> Tuple[np.array, np.array]:
        """
        Generate bivariate geometric brownian motion
        @param st_1: asset 1 stock price
        @param iv_1: asset 1 implied volatility
        @param d_1: asset 1 continuously compounded dividend yield
        @param b_1: asset 1 continuously compounded repo rate or borrowing cost
        @param st_2: asset 2 stock price
        @param iv_2: asset 2 implied volatility
        @param d_2: asset 2 continuously compounded dividend yield
        @param b_2: asset 2 continuously compounded repo rate or borrowing cost
        @param rho: implied correlation between asset 1 and asset 2
        @param r: continuously compounded risk-free interest rate
        @return: 2 correlated assets generated timeseries
        """
        dt = 1 / self.num_path_per_year
        mu = np.array([0, 0])
        cov = np.array([[1, rho], [rho, 1]])

        ts_1 = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st_1)
        ts_2 = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st_2)

        z = np.random.multivariate_normal(mean=mu, cov=cov, size=(self.num_path, self.num_simulation))

        drift_1 = r - d_1 - b_1
        drift_2 = r - d_2 - b_2

        for i in range(1, self.num_path + 1):
            ts_1[i] = ts_1[i - 1] * np.exp((drift_1 - 0.5 * iv_1 ** 2) * dt + iv_1 * np.sqrt(dt) * z[i - 1, :, 0])
            ts_2[i] = ts_2[i - 1] * np.exp((drift_2 - 0.5 * iv_2 ** 2) * dt + iv_2 * np.sqrt(dt) * z[i - 1, :, 1])

        return ts_1, ts_2


class Priceable(ABC):
    def __init__(self):
        self.pv: Union[None, float] = None

    @abstractmethod
    def calculate_present_value(self, pricing_model: Union[None, str]) -> None:
        ...

    def get_present_value(self) -> float:
        return self.pv


class Forward(Priceable):

    def __init__(self, st: float, k: float, q: float, b: float, r: float, t: float):
        """
        Initialize a Forward object
        @param st: stock price
        @param k: strike price
        @param q: continuously compounded dividend yield
        @param b: continuously compounded repo rate or borrowing cost
        @param r: continuously compounded risk-free interest rate
        @param t: time in fraction of a year
        """
        super().__init__()
        self.st = st
        self.k = k
        self.q = q
        self.b = b
        self.r = r
        self.t = t
        self.fwd = None
        self.d_strike = None
        self.d_fwd = None

    def __post_init__(self):
        self.d = np.exp(-self.r * self.t)

    def calculate_present_value(self, pricing_model=None) -> None:
        self.__post_init__()
        self.fwd = self.st * np.exp((self.r - self.q - self.b) * self.t)
        self.d_strike = self.k * self.d
        self.d_fwd = self.fwd * self.d
        self.pv = self.d_fwd - self.d_strike


class OptionPricing(Priceable):

    def __init__(self, st: float, k: float, iv: float, q: float, b: float, r: float, t: float, kind: str):
        """
        Initialize a OptionPricing object
        @param st: stock price
        @param k: strike price
        @param iv: implied volatility
        @param q: continuously compounded dividend yield
        @param b: continuously compounded repo rate or borrowing cost
        @param r: continuously compounded risk-free interest rate
        @param t: time in fraction of a year
        @param kind: 'call' for a call option, 'put' for a put option

        """
        super().__init__()
        self.st = st
        self.k = k
        self.iv = iv
        self.q = q
        self.b = b
        self.r = r
        self.t = t
        self.kind = kind
        self.__post_init__()

    def __post_init__(self):
        self.fwd = self.st * np.exp((self.r - self.q - self.b) * self.t)
        self.d = np.exp(-self.r * self.t)
        self.d_fwd = self.fwd * self.d
        self.scaled_iv = self.iv * np.sqrt(self.t)

    @staticmethod
    def _log_normal_pdf(x: float, mu: float, sigma: float) -> float:
        """
        Calculate log-normal probability density
        @param x: random variable
        @param mu: log-normal distribution mean
        @param sigma: log-normal distribution sigma
        @return: probability density
        """
        exponent = -((np.log(x) - mu) ** 2) / (2 * sigma ** 2)
        coefficient = 1 / (x * sigma * np.sqrt(2 * np.pi))
        pdf = coefficient * np.exp(exponent)
        return pdf

    def _integrand(self, st, k, mu, sigma, payoff) -> float:
        """
        Calculate option intrinsic value weighted by the log-normal density
        @param st: stock price
        @param k: strike price
        @param mu: mean parameter of log-normal distribution
        @param sigma: standard-deviation parameter of log-normal distribution
        @param payoff: 'call' for a call option, 'put' for a put option
        @return: payoff * density
        """
        if payoff == 'call':
            intrinsic_value = max(0.0, (st - k))
        elif payoff == 'put':
            intrinsic_value = max(0.0, (k - st))
        else:
            raise ValueError
        return intrinsic_value * self._log_normal_pdf(x=st, mu=mu, sigma=sigma)

    def numerical_integration(self) -> None:
        self.__post_init__()

        mu = np.log(self.d_fwd) + (self.r - 0.5 * self.iv ** 2) * self.t

        if self.kind == 'call':
            l_bound = self.k
            u_bound = np.inf
        elif self.kind == 'put':
            l_bound = 0
            u_bound = self.k
        else:
            raise ValueError

        self.pv = quad(self._integrand, l_bound, u_bound, args=(self.k, mu, self.scaled_iv, self.kind))[0] * self.d

    def black_scholes(self) -> None:
        self.__post_init__()

        if self.t > 0:
            d1 = ((np.log(self.d_fwd / self.k) + (self.r + 0.5 * self.iv ** 2) * self.t) / self.scaled_iv)
            d2 = d1 - self.scaled_iv
            if self.kind == 'call':
                self.pv = self.d_fwd * norm.cdf(d1) - self.k * self.d * norm.cdf(d2)
            elif self.kind == 'put':
                self.pv = self.k * self.d * norm.cdf(-d2) - self.d_fwd * norm.cdf(-d1)
            else:
                raise ValueError
        else:
            if self.kind == 'call':
                self.pv = max(0.0, self.st - self.k)
            elif self.kind == 'put':
                self.pv = max(0.0, self.k - self.st)
            else:
                raise ValueError

    def calculate_present_value(self, pricing_model: str) -> None:
        """
        Calculate present discounted value
        @param pricing_model: 'BlackScholes' or 'NumericalIntegration'
        """
        if pricing_model == "BlackScholes":
            self.black_scholes()
        elif pricing_model == 'NumericalIntegration':
            self.numerical_integration()
        else:
            raise TypeError
