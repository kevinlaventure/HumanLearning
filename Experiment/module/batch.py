import sys
sys.path.append('/Users/kevinlaventure/Project/HumanLearning/Experiment/module')

import pandas as pd
from tqdm.auto import tqdm
from typing import Callable
from pricing import OptionPricer, DualDigitalPricer


class Batch:

    def __init__(self, pricer: Callable, extra_output_function: str, display_progress: bool):
        self.batch_data = dict()
        self.pricer = pricer
        self.extra_output_function = extra_output_function
        self.display_progress = display_progress

    @staticmethod
    def _build_rolling_timeseries(timeseries: pd.DataFrame, roll: int) -> pd.DataFrame:
        raw_timeseries = timeseries.copy()
        reversed_index = raw_timeseries.index[::-1].to_list()
        raw_priceable_data = raw_timeseries.to_dict(orient='index')

        roll_id = 0
        time_to_roll = 0

        reshaped_priceable_data = dict()
        for i in reversed_index:
            reshaped_priceable_data[(i, roll_id)] = {**raw_priceable_data[i]}
            reshaped_priceable_data[(i, roll_id)]['time_to_roll'] = time_to_roll

            time_to_roll += 1

            if time_to_roll > roll:
                time_to_roll = 0
                roll_id += 1

                reshaped_priceable_data[(i, roll_id)] = raw_priceable_data.get(i)
                reshaped_priceable_data[(i, roll_id)]['time_to_roll'] = time_to_roll

                time_to_roll += 1

        priceable_data = pd.DataFrame(reshaped_priceable_data).transpose().iloc[::-1]
        priceable_data = priceable_data.reset_index(names=['date', 'roll_id'])
        batch_start = priceable_data[priceable_data['time_to_roll'] == roll].index[0]
        priceable_data = priceable_data.iloc[batch_start:]
        priceable_data.loc[:, 'roll_id'] = abs(priceable_data['roll_id'] - priceable_data['roll_id'].max())
        priceable_data = priceable_data.set_index(['date', 'roll_id'])
        return priceable_data

    def calculate_batch(self) -> pd.DataFrame:

        dates = list(self.batch_data.keys())

        for date in tqdm(dates, display=self.display_progress):

            priceable = self.pricer(**self.batch_data.get(date))

            priceable.calculate_present_value()
            pv = priceable.get_present_value()
            self.batch_data[date]['pv'] = pv

            if self.extra_output_function is not None:
                getattr(priceable, self.extra_output_function)()
                greeks = priceable.get_greeks()
                for key in greeks:
                    self.batch_data[date][key] = greeks.get(key)

        return pd.DataFrame.from_dict(self.batch_data, orient='index')


class OptionBatch(Batch):

    def __init__(self,  extra_output_function: str, display_progress: bool):
        super().__init__(pricer=OptionPricer, extra_output_function=extra_output_function,
                         display_progress=display_progress)

    def build_static_rolling_batch(self,
                                   timeseries: pd.DataFrame, roll: int,
                                   k: float, iv: float, q: float, b: float, r: float, t: int,
                                   kind: str, model: str):
        rolling_batch = self._build_rolling_timeseries(timeseries=timeseries, roll=roll)
        fixing = rolling_batch.loc[rolling_batch['time_to_roll'] == roll, 'st'] * k
        rolling_batch.loc[rolling_batch['time_to_roll'] == roll, 'k'] = fixing
        rolling_batch.loc[:, 'k'] = rolling_batch['k'].ffill()
        rolling_batch.loc[:, ['iv', 'q', 'b', 'r']] = (iv, q, b, r)
        rolling_batch.loc[:, 'kind'] = kind
        rolling_batch.loc[:, 't'] = (t - (roll - rolling_batch['time_to_roll'])) / 252
        rolling_batch.loc[:, 'model'] = model
        rolling_batch = rolling_batch.set_index(keys='time_to_roll', append=True)
        self.batch_data = rolling_batch.to_dict(orient='index')


class DualDigitalBatch(Batch):

    def __init__(self, extra_output_function: str, display_progress: bool):
        super().__init__(pricer=DualDigitalPricer, extra_output_function=extra_output_function,
                         display_progress=display_progress)

    def build_static_rolling_batch(self,
                                   timeseries: pd.DataFrame, roll: int,
                                   k1: float, iv1: float, q1: float, b1: float, direction1: str,
                                   k2: float, iv2: float, q2: float, b2: float, direction2: str,
                                   rho: float, r: float, t: int, notional: int, model: str) -> None:
        rolling_batch = self._build_rolling_timeseries(timeseries=timeseries, roll=roll)
        fixing1 = rolling_batch.loc[rolling_batch['time_to_roll'] == roll, 'st1'] * k1
        fixing2 = rolling_batch.loc[rolling_batch['time_to_roll'] == roll, 'st2'] * k2
        rolling_batch.loc[rolling_batch['time_to_roll'] == roll, 'k1'] = fixing1
        rolling_batch.loc[rolling_batch['time_to_roll'] == roll, 'k2'] = fixing2
        rolling_batch.loc[:, ['k1', 'k2']] = rolling_batch[['k1', 'k2']].ffill()
        rolling_batch.loc[:, ['iv1', 'iv2', 'q1', 'q2', 'b1', 'b2']] = (iv1, iv2, q1, q2, b1, b2)
        rolling_batch.loc[:, ['direction1', 'direction2']] = (direction1, direction2)
        rolling_batch.loc[:, ['rho', 'r', 'notional']] = (rho, r, notional)
        rolling_batch.loc[:, 't'] = (t - (roll - rolling_batch['time_to_roll'])) / 252
        rolling_batch.loc[:, 'model'] = model
        rolling_batch = rolling_batch.set_index(keys='time_to_roll', append=True)
        self.batch_data = rolling_batch.to_dict(orient='index')



