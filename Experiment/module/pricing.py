import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Tuple
from abc import ABC, abstractmethod
from scipy.integrate import quad, nquad


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

        ts = np.full(shape=(self.num_path + 1, self.num_simulation), fill_value=st, dtype=float)

        z = np.random.standard_normal(size=(self.num_path, self.num_simulation))

        drift = r - d - b
        for i in range(1, self.num_path + 1):
            ts[i] = ts[i - 1] * np.exp((drift - 0.5 * iv ** 2) * dt + iv * np.sqrt(dt) * z[i - 1])

        return ts

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

        for i in range(1, self.num_path + 1):
            ts_1[i] = ts_1[i - 1] * np.exp((drift_1 - 0.5 * iv1 ** 2) * dt + iv1 * np.sqrt(dt) * z[i - 1, :, 0])
            ts_2[i] = ts_2[i - 1] * np.exp((drift_2 - 0.5 * iv2 ** 2) * dt + iv2 * np.sqrt(dt) * z[i - 1, :, 1])

        return ts_1, ts_2


class Priceable(ABC):
    def __init__(self):
        self.pv: Union[None, float] = None
        self.greeks = dict()
        self.extra_output = dict()

    @abstractmethod
    def calculate_present_value(self) -> None:
        ...

    @abstractmethod
    def calculate_greeks(self) -> None:
        ...

    def calculate_derivative(self, parameter1: str, parameter2: Union[None, str], dx: float = 0.01) -> float:

        # -------
        # RETRIEVE INITIAL PV
        # -------
        if self.pv is None:
            self.calculate_present_value()
        init_pv = self.pv

        # -------
        # COMPUTE FIRST ORDER DERIVATIVE
        # -------
        parm1_init_value = self.__dict__[parameter1]
        self.__dict__[parameter1] = self.__dict__[parameter1] + dx
        self.calculate_present_value()
        param1_bumped_pv = self.pv
        self.__dict__[parameter1] = parm1_init_value
        df = param1_bumped_pv - init_pv
        first_order_derivative = df / dx
        result = first_order_derivative

        # -------
        # COMPUTE SECOND ORDER DERIVATIVE
        # -------
        if parameter2 is not None and parameter2 == parameter1:
            parm2_init_value = self.__dict__[parameter2]
            self.__dict__[parameter2] = self.__dict__[parameter2] - dx
            self.calculate_present_value()
            param2_bumped_pv = self.pv
            self.__dict__[parameter2] = parm2_init_value
            df = param1_bumped_pv - (2 * init_pv) + param2_bumped_pv
            second_order_derivative = df/(dx**2)
            result = second_order_derivative
        elif parameter2 is not None and parameter2 != parameter1:
            parm2_init_value = self.__dict__[parameter2]
            self.__dict__[parameter2] = self.__dict__[parameter2] + dx
            self.calculate_present_value()
            param2_bumped_pv = self.pv
            self.__dict__[parameter1] = self.__dict__[parameter1] + dx
            self.calculate_present_value()
            param1_param2_bumped_pv = self.pv
            df = param1_param2_bumped_pv - param2_bumped_pv - param1_bumped_pv + init_pv
            x_second_order_derivative = df/(dx**2)
            result = x_second_order_derivative
            self.__dict__[parameter1] = parm1_init_value
            self.__dict__[parameter2] = parm2_init_value

        # -------
        # RESET PV
        # -------
        self.pv = init_pv
        return result

    def get_present_value(self) -> float:
        return self.pv

    def get_greeks(self):
        return self.greeks


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

    def __post_init__(self):
        self.d = np.exp(-self.r * self.t)
        self.fwd = self.st * np.exp((self.r - self.q - self.b) * self.t)

    def calculate_present_value(self, pricing_model=None) -> None:
        self.__post_init__()
        self.pv = (self.fwd - self.k) * self.d

    def calculate_greeks(self) -> None:
        self.greeks['dst'] = self.calculate_derivative(parameter1='st', parameter2=None)
        self.greeks['dst**2'] = self.calculate_derivative(parameter1='st', parameter2='st')
        self.greeks['dk'] = self.calculate_derivative(parameter1='k', parameter2=None)
        self.greeks['dq'] = self.calculate_derivative(parameter1='q', parameter2=None) * 0.01
        self.greeks['db'] = self.calculate_derivative(parameter1='b', parameter2=None) * 0.01
        self.greeks['dr'] = self.calculate_derivative(parameter1='r', parameter2=None) * 0.01
        self.greeks['dt'] = self.calculate_derivative(parameter1='t', parameter2=None) * -(1/252)


class OptionPricing(Priceable):

    def __init__(self, st: float, k: float, iv: float, q: float, b: float, r: float, t: float, kind: str, model: str):
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
        @param model: 'black_scholes' or 'numerical_integration'
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
        self.model = model

    def __post_init__(self):
        self.fwd = self.st * np.exp((self.r - self.q - self.b) * self.t)
        self.d = np.exp(-self.r * self.t)
        self.scaled_iv = self.iv * np.sqrt(self.t)

    def calculate_intrinsic_value(self) -> None:
        if self.kind == 'call':
            self.pv = max(0.0, self.st - self.k)
        elif self.kind == 'put':
            self.pv = max(0.0, self.k - self.st)

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

    def _integrand(self, st: float, k: float, mu: float, sigma: float, payoff: str) -> float:
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

        if self.t > 0:
            self.__post_init__()
            d_fwd = self.fwd * self.d
            mu = np.log(d_fwd) + (self.r - 0.5 * self.iv ** 2) * self.t
            if self.kind == 'call':
                l_bound = self.k
                u_bound = np.inf
            elif self.kind == 'put':
                l_bound = 0
                u_bound = self.k
            else:
                raise ValueError
            self.pv = quad(self._integrand, l_bound, u_bound, args=(self.k, mu, self.scaled_iv, self.kind))[0] * self.d
        else:
            self.calculate_intrinsic_value()

    def black_scholes(self) -> None:
        self.__post_init__()
        if self.t > 0:
            d1 = ((np.log(self.fwd / self.k) + (0.5 * self.iv ** 2) * self.t) / self.scaled_iv)
            d2 = d1 - self.scaled_iv
            if self.kind == 'call':
                self.pv = self.d * (self.fwd * norm.cdf(d1) - self.k * norm.cdf(d2))
            elif self.kind == 'put':
                self.pv = self.d * (self.k * norm.cdf(-d2) - self.fwd * norm.cdf(-d1))
            else:
                raise ValueError
        else:
            self.calculate_intrinsic_value()

    def calculate_present_value(self) -> None:
        """
        Calculate present discounted value
        """
        getattr(self, self.model)()

    def calculate_greeks(self) -> None:
        if self.t > 0:
            self.greeks['dst'] = self.calculate_derivative(parameter1='st', parameter2=None)
            self.greeks['dst**2'] = self.calculate_derivative(parameter1='st', parameter2='st')
            self.greeks['dk'] = self.calculate_derivative(parameter1='k', parameter2=None)
            self.greeks['div'] = self.calculate_derivative(parameter1='iv', parameter2=None) * 0.01
            self.greeks['dq'] = self.calculate_derivative(parameter1='q', parameter2=None) * 0.01
            self.greeks['db'] = self.calculate_derivative(parameter1='b', parameter2=None) * 0.01
            self.greeks['dr'] = self.calculate_derivative(parameter1='r', parameter2=None) * 0.01
            self.greeks['dt'] = self.calculate_derivative(parameter1='t', parameter2=None) * -(1/252)
        else:
            self.greeks['dst'] = np.nan
            self.greeks['dst**2'] = np.nan
            self.greeks['dk'] = np.nan
            self.greeks['div'] = np.nan
            self.greeks['dq'] = np.nan
            self.greeks['db'] = np.nan
            self.greeks['dr'] = np.nan
            self.greeks['dt'] = np.nan


class DualDigital(Priceable):
    NUM_SIMULATION = 15_000
    NUM_PATH_PER_YEAR = 252

    def __init__(self,
                 st1: float, k1: float, iv1: float, q1: float, b1: float, direction1: str,
                 st2: float, k2: float, iv2: float, q2: float, b2: float, direction2: str,
                 rho: float, r: float, t: float, unit: int, model: str):
        super().__init__()
        self.st1 = st1
        self.k1 = k1
        self.iv1 = iv1
        self.q1 = q1
        self.b1 = b1
        self.direction1 = direction1
        self.st2 = st2
        self.k2 = k2
        self.iv2 = iv2
        self.q2 = q2
        self.b2 = b2
        self.direction2 = direction2
        self.rho = rho
        self.r = r
        self.t = t
        self.unit = unit
        self.model = model

    def __post_init__(self):
        self.fwd1 = self.st1 * np.exp((self.r - self.q1 - self.b1) * self.t)
        self.fwd2 = self.st2 * np.exp((self.r - self.q2 - self.b2) * self.t)
        self.d = np.exp(-self.r * self.t)

    def calculate_intrinsic_value(self):
        if self.direction1 == 'up':
            if self.st1 > self.k1:
                r1 = 1
            else:
                r1 = 0
        elif self.direction1 == 'down':
            if self.st1 < self.k1:
                r1 = 1
            else:
                r1 = 0
        else:
            raise ValueError

        if self.direction2 == 'up':
            if self.st2 > self.k2:
                r2 = 1
            else:
                r2 = 0
        elif self.direction2 == 'down':
            if self.st2 < self.k2:
                r2 = 1
            else:
                r2 = 0
        else:
            raise ValueError

        r3 = r1 + r2
        if r3 == 2:
            self.pv = self.unit
        else:
            self.pv = 0

    @staticmethod
    def _bivariate_log_normal_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho):
        term1 = 1 / (2*np.pi*sigma_x*sigma_y*np.sqrt(1 - rho**2)*x*y)
        term2 = - (1 / (2 * (1 - rho**2)))
        term3 = ((np.log(x) - mu_x) / sigma_x)**2
        term4 = ((np.log(y) - mu_y) / sigma_y)**2
        term5 = -2*rho*((np.log(x) - mu_x) / sigma_x)*((np.log(y) - mu_y) / sigma_y)
        term6 = np.exp(term2*(term3+term4+term5))
        return term1 * term6

    def _integrand(self, st1: float, st2: float, mu1: float, mu2: float, iv1: float, iv2: float, rho: float, unit: int):
        return unit * self._bivariate_log_normal_pdf(st1, st2, mu1, mu2, iv1, iv2, rho)

    def numerical_integration(self):
        if self.t > 0:
            self.__post_init__()
            d_fwd1 = self.fwd1 * self.d
            d_fwd2 = self.fwd2 * self.d
            mu1 = np.log(d_fwd1) + (self.r - 0.5 * self.iv1 ** 2) * self.t
            mu2 = np.log(d_fwd2) + (self.r - 0.5 * self.iv2 ** 2) * self.t
            iv1 = self.iv1 * np.sqrt(self.t)
            iv2 = self.iv2 * np.sqrt(self.t)
            bound1 = [self.k1, np.inf] if self.direction1 == 'up' else [0, self.k1]
            bound2 = [self.k2, np.inf] if self.direction2 == 'up' else [0, self.k2]
            e_payoff = nquad(self._integrand, ranges=[bound1, bound2], args=(mu1, mu2, iv1, iv2, self.rho, self.unit))[0]
            self.pv = e_payoff * self.d
        else:
            self.calculate_intrinsic_value()

    def montecarlo(self):
        self.__post_init__()
        num_path = int(self.NUM_PATH_PER_YEAR * self.t)
        mc = MonteCarlo(num_path=num_path, num_simulation=self.NUM_SIMULATION, num_path_per_year=self.NUM_PATH_PER_YEAR)
        ts_1, ts_2 = mc.bivariate_gbm(
            st1=self.st1, iv1=self.iv1, q1=self.q1, b1=self.b1,
            st2=self.st1, iv2=self.iv1, q2=self.q1, b2=self.b1,
            rho=self.rho, r=self.r)

        df = pd.DataFrame([pd.DataFrame(ts_1).iloc[-1], pd.DataFrame(ts_2).iloc[-1]]).transpose()
        df.columns = ['ts_1', 'ts_2']
        if self.direction1 == 'up':
            df['direction1'] = (df['ts_1'] > self.k1).astype(int)
        elif self.direction1 == 'down':
            df['direction1'] = (df['ts_1'] < self.k1).astype(int)
        if self.direction2 == 'up':
            df['direction2'] = (df['ts_2'] > self.k2).astype(int)
        elif self.direction1 == 'down':
            df['direction2'] = (df['ts_2'] < self.k2).astype(int)

        df['direction1 & direction2'] = ((df['direction1'] + df['direction2']) == 2).astype(int)
        self.extra_output['individuals'] = (df['direction1'].sum() / df.shape[0], df['direction2'].sum() / df.shape[0])
        proba = df['direction1 & direction2'].sum() / df.shape[0]
        self.pv = proba * self.d
        return df

    def calculate_present_value(self) -> None:
        """
        Calculate present discounted value
        """
        getattr(self, self.model)()

    def calculate_greeks(self) -> None:
        if self.t > 0:
            self.greeks['dst1'] = self.calculate_derivative(parameter1='st1', parameter2=None)
            self.greeks['dst2'] = self.calculate_derivative(parameter1='st2', parameter2=None)
            self.greeks['dst1**2'] = self.calculate_derivative(parameter1='st1', parameter2='st1')
            self.greeks['dst2**2'] = self.calculate_derivative(parameter1='st2', parameter2='st2')
            self.greeks['dst1*dst2'] = self.calculate_derivative(parameter1='st1', parameter2='st2')
            self.greeks['dt'] = self.calculate_derivative(parameter1='t', parameter2=None) * -(1/252)
        else:
            self.greeks['dst1'] = np.nan
            self.greeks['dst2'] = np.nan
            self.greeks['dst1**2'] = np.nan
            self.greeks['dst2**2'] = np.nan
            self.greeks['dst1*dst2'] = np.nan
            self.greeks['dt'] = np.nan

    def calculate_x_gamma(self) -> None:
        if self.t > 0:
            self.greeks['dst1*dst2'] = self.calculate_derivative(parameter1='st1', parameter2='st2')
        else:
            self.greeks['dst1*dst2'] = np.nan

    def calculate_delta(self) -> None:
        if self.t > 0:
            self.greeks['dst1'] = self.calculate_derivative(parameter1='st1', parameter2=None)
            self.greeks['dst2'] = self.calculate_derivative(parameter1='st2', parameter2=None)
        else:
            self.greeks['dst1'] = np.nan
            self.greeks['dst2'] = np.nan


mc = MonteCarlo(num_path=252, num_simulation=100)
x = mc.univariate_gbm(st=100, iv=0.2, d=0, b=0, r=0)