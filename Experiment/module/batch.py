import sys

sys.path.append('/Users/kevinlaventure/Project/HumanLearning/Experiment/module')

import pandas as pd
from typing import Union
from tqdm.auto import tqdm


class RollingBatchBuilder:

    def __init__(self, timeseries: pd.DataFrame, roll: int):
        """
        Initialize a RollingBatchBuilder object
        :param timeseries: uni or multidimensional data
        :param roll: number of step before roll
        """
        self.raw_timeseries = timeseries
        self.roll = roll
        self.reversed_index = self.raw_timeseries.index[::-1].to_list()
        self.raw_priceable_data = self.raw_timeseries.to_dict(orient='index')

    def _build_rolling_batch(self):
        roll_id = 0
        time_to_roll = 0

        reshaped_priceable_data = dict()
        for i in self.reversed_index:
            reshaped_priceable_data[(i, roll_id)] = {**self.raw_priceable_data[i]}
            reshaped_priceable_data[(i, roll_id)]['time_to_roll'] = time_to_roll

            time_to_roll += 1

            if time_to_roll > self.roll:

                time_to_roll = 0
                roll_id += 1

                reshaped_priceable_data[(i, roll_id)] = self.raw_priceable_data.get(i)
                reshaped_priceable_data[(i, roll_id)]['time_to_roll'] = time_to_roll

                time_to_roll += 1

        priceable_data = pd.DataFrame(reshaped_priceable_data).transpose().iloc[::-1]
        priceable_data = priceable_data.reset_index(names=['date', 'roll_id'])
        batch_start = priceable_data[priceable_data['time_to_roll'] == self.roll].index[0]
        priceable_data = priceable_data.iloc[batch_start:]
        priceable_data.loc[:, 'roll_id'] = abs(priceable_data['roll_id'] - priceable_data['roll_id'].max())
        priceable_data = priceable_data.set_index(['date', 'roll_id'])
        return priceable_data


class DualDigitalBatchBuilder(RollingBatchBuilder):

    def __init__(self, timeseries: pd.DataFrame, roll: int):
        """
        Initialize a DualDigitalBatchBuilder object
        :param timeseries: uni or multidimensional data
        :param roll: number of step before roll
        """
        super().__init__(timeseries, roll)

    def build_batch(self,
                    k1: float, iv1: float, q1: float, b1: float, direction1: str,
                    k2: float, iv2: float, q2: float, b2: float, direction2: str,
                    rho: float, r: float, t: float, notional: int, model: str) -> pd.DataFrame:

        rolling_batch = self._build_rolling_batch()
        fixing1 = rolling_batch.loc[rolling_batch['time_to_roll'] == self.roll, 'st1'] * k1
        fixing2 = rolling_batch.loc[rolling_batch['time_to_roll'] == self.roll, 'st2'] * k2
        rolling_batch.loc[rolling_batch['time_to_roll'] == self.roll, 'k1'] = fixing1
        rolling_batch.loc[rolling_batch['time_to_roll'] == self.roll, 'k2'] = fixing2
        rolling_batch.loc[:, ['k1', 'k2']] = rolling_batch[['k1', 'k2']].ffill()
        rolling_batch.loc[:, ['iv1', 'iv2', 'q1', 'q2', 'b1', 'b2']] = (iv1, iv2, q1, q2, b1, b2)
        rolling_batch.loc[:, ['direction1', 'direction2']] = (direction1, direction2)
        rolling_batch.loc[:, ['rho', 'r', 'notional']] = (rho, r, notional)
        rolling_batch.loc[:, 't'] = rolling_batch['time_to_roll'] / 252
        rolling_batch.loc[:, 'model'] = model
        rolling_batch = rolling_batch.drop('time_to_roll', axis=1)
        return rolling_batch


class BatchPricing:

    def __init__(self, batch: pd.DataFrame):

        self.raw_batch = batch
        self.__post_init__()

    def __post_init__(self):
        self.batch = self.raw_batch.to_dict(orient='index')

    def calculate(self, pricer, greeks_to_calculate: Union[None, str] = None):

        self.__post_init__()
        dates = list(self.batch.keys())

        for date in tqdm(dates):

            priceable = pricer(**self.batch.get(date))

            priceable.calculate_present_value()
            pv = priceable.get_present_value()

            if greeks_to_calculate is not None:
                getattr(priceable, greeks_to_calculate)()
                greeks = priceable.get_greeks()
                self.batch[date]['pv'] = pv
                for key in greeks:
                    self.batch[date][key] = greeks.get(key)
        return self.batch


# -----
# IMPORT DATA
# -----
df1 = pd.read_csv('/Users/kevinlaventure/Project/HumanLearning/Experiment/data/SPY.csv', index_col=0, parse_dates=True).loc[:, ['Adj Close']]
df2 = pd.read_csv('/Users/kevinlaventure/Project/HumanLearning/Experiment/data/IWM.csv', index_col=0, parse_dates=True).loc[:, ['Adj Close']]
df = pd.concat([df1, df2], axis=1).dropna()
df.columns = ['st1', 'st2']
df_snap = df.iloc[-21:]

bb_obj = DualDigitalBatchBuilder(timeseries=df_snap, roll=5)
bb_obj._build_rolling_batch()