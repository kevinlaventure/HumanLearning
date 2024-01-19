import sys

sys.path.append('/Users/kevinlaventure/Project/HumanLearning/Experiment/module')

import pandas as pd
from copy import copy
from typing import Union
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from pricing import DualDigitalPricer


class Backtest(ABC):
    def __init__(self, periods_per_year: int = 252, display_bt_progress: bool = False):
        self.periods_per_year = periods_per_year
        self.display_bt_progress = display_bt_progress

    @abstractmethod
    def _calculate_iterator(self, priceable_data, compute_extra_output, display_progress):
        ...

    @abstractmethod
    def calculate_backtest(self, **kwargs):
        ...

    @abstractmethod
    def get_results(self):
        ...


class DualDigitalBacktest(Backtest):

    def __init__(self, periods_per_year: int = 252, display_bt_progress: bool = False):
        super().__init__(periods_per_year, display_bt_progress)
        self.periods_per_year = periods_per_year
        self.display_bt_progress = display_bt_progress

    @staticmethod
    def _compute_cum_pnl(df):
        pnl_col = list(filter(lambda x: 'pnl' in x, df.columns))
        for col in pnl_col:
            df.loc[:, f'cum_{col}'] = df[col].cumsum()

    def _calculate_iterator(self,
                            priceable_data: dict,
                            compute_extra_output: bool = False,
                            display_bt_progress: bool = False) -> pd.DataFrame:

        # -------
        # COMPUTE PRODUCT PV AND GREEKS TIMESERIES
        # -------

        periods = list(priceable_data.keys())
        bt_data = copy(priceable_data)

        for date in tqdm(periods, display=display_bt_progress):

            # -------
            # COMPUTE PRODUCT PV
            # -------

            priceable = DualDigitalPricer(**priceable_data.get(date))

            priceable.calculate_present_value()
            pv = priceable.get_present_value()

            # -------
            # COMPUTE PRODUCT GREEKS
            # -------

            if compute_extra_output:
                priceable.calculate_greeks()
            else:
                priceable.calculate_delta()
            greeks = priceable.get_greeks()

            bt_data[date]['pv'] = pv
            bt_data[date]['delta_st1'] = greeks.get('dst1')
            bt_data[date]['delta_st2'] = greeks.get('dst2')

            if compute_extra_output:
                bt_data[date]['gamma_st1'] = greeks.get('dst1**2')
                bt_data[date]['gamma_st2'] = greeks.get('dst2**2')
                bt_data[date]['x_gamma'] = greeks.get('dst1*dst2')
                bt_data[date]['theta'] = greeks.get('dt')

        # -------
        # COMPUTE PNL BREAK-DOWN
        # -------

        bt_data_df = pd.DataFrame.from_dict(bt_data, orient='index')

        bt_data_df.loc[:, 'option_pnl'] = bt_data_df['pv'].diff()

        bt_data_df.loc[:, 'delta_st1_pnl'] = bt_data_df['st1'].diff() * bt_data_df['delta_st1'].shift()
        bt_data_df.loc[:, 'delta_st2_pnl'] = bt_data_df['st2'].diff() * bt_data_df['delta_st2'].shift()
        bt_data_df.loc[:, 'delta_pnl'] = bt_data_df['delta_st1_pnl'] + bt_data_df['delta_st2_pnl']

        if compute_extra_output:
            bt_data_df.loc[:, 'gamma_st1_pnl'] = bt_data_df['st1'].diff().pow(2) * bt_data_df['gamma_st1'].shift() * 0.5
            bt_data_df.loc[:, 'gamma_st2_pnl'] = bt_data_df['st2'].diff().pow(2) * bt_data_df['gamma_st2'].shift() * 0.5

            bt_data_df.loc[:, 'gamma_pnl'] = bt_data_df['gamma_st1_pnl'] + bt_data_df['gamma_st2_pnl']

            bt_data_df.loc[:, 'x_gamma_pnl'] = bt_data_df['st1'].diff() * bt_data_df['st2'].diff() * bt_data_df[
                'x_gamma'].shift()

            bt_data_df.loc[:, 'explained_pnl'] = bt_data_df['delta_pnl']
            bt_data_df.loc[:, 'explained_pnl'] += bt_data_df['gamma_pnl']
            bt_data_df.loc[:, 'explained_pnl'] += bt_data_df['x_gamma_pnl']
            bt_data_df.loc[:, 'explained_pnl'] += bt_data_df['theta'].shift()
            self._compute_cum_pnl(bt_data_df)

        return bt_data_df

    def calculate_backtest(self,
                           df: pd.DataFrame,
                           roll_period: Union[None, int],
                           k1_pct: float,
                           k2_pct: float,
                           rolling_window: Union[None, int] = None):
        pass

    def get_results(self):
        pass

