import sys
sys.path.append('/Users/kevinlaventure/Project/HumanLearning/Experiment/module')
import pandas as pd
from typing import Union
from tqdm.auto import tqdm

from pricing import DualDigital


class Backtest:

    def __init__(self, model: Union[None, str], periods_per_year: int = 252):
        self.model = model
        self.periods_per_year = periods_per_year

    def dual_digital(self,
                     data: dict,
                     k1: float, k2: float,
                     iv1: float, iv2: float,
                     q1: float, q2: float,
                     b1: float, b2: float,
                     rho: float, r: float, unit: int):

        # -------
        # COMPUTE PRODUCT PV AND GREEKS TIMESERIES
        # -------

        bt = dict()

        dates = list(data.keys())
        date_t0 = dates[0]

        st1_t0 = data.get(date_t0).get('st1')
        st2_t0 = data.get(date_t0).get('st2')

        business_day_to_expiry = len(dates) - 1
        t = business_day_to_expiry / self.periods_per_year

        for date in tqdm(dates):

            st1 = data.get(date).get('st1')
            st2 = data.get(date).get('st2')

            st1_pct = st1 / st1_t0
            st2_pct = st2 / st2_t0

            priceable_up = DualDigital(
                st1=st1_pct, k1=k1, iv1=iv1, q1=q1, b1=b1, direction1='up',
                st2=st2_pct, k2=k2, iv2=iv2, q2=q2, b2=b2, direction2='up',
                rho=rho, r=r, t=t, unit=unit, model=self.model
            )

            priceable_up.calculate_present_value()
            priceable_up.calculate_delta()
            pv_up = priceable_up.get_present_value()
            greeks_up = priceable_up.get_greeks()

            priceable_down = DualDigital(
                st1=st1_pct, k1=k1, iv1=iv1, q1=q1, b1=b1, direction1='down',
                st2=st2_pct, k2=k2, iv2=iv2, q2=q2, b2=b2, direction2='down',
                rho=rho, r=r, t=t, unit=unit, model=self.model
            )

            priceable_down.calculate_present_value()
            priceable_down.calculate_delta()
            pv_down = priceable_down.get_present_value()
            greeks_down = priceable_down.get_greeks()

            pv = pv_up + pv_down
            delta_st1 = (greeks_up.get('dst1') + greeks_down.get('dst1')) / st1
            delta_st2 = (greeks_up.get('dst2') + greeks_down.get('dst2')) / st2
            # gamma_st1 = (greeks_up.get('dst1**2') + greeks_down.get('dst1**2')) / (st1 ** 2)
            # gamma_st2 = (greeks_up.get('dst2**2') + greeks_down.get('dst2**2')) / (st2 ** 2)
            # x_gamma = (greeks_up.get('dst1*dst2') + greeks_down.get('dst1*dst2')) / (st1 * st2)
            # theta = greeks_up.get('dt') + greeks_down.get('dt')

            bt[date] = {
                'business_day_to_expiry': business_day_to_expiry,
                't': t,
                'st1': st1, 'st2': st2,
                'st1_pct': st1_pct, 'st2_pct': st2_pct,
                'pv': pv,
                'delta_st1': delta_st1, 'delta_st2': delta_st2,
                # 'gamma_st1': gamma_st1, 'gamma_st2': gamma_st2,
                # 'x_gamma': x_gamma, 'theta': theta
            }

            business_day_to_expiry = business_day_to_expiry - 1
            t = business_day_to_expiry / self.periods_per_year

        # -------
        # COMPUTE REPLICATION STRATEGY AND PNL BREAK-DOWN
        # -------
        bt_df = pd.DataFrame.from_dict(bt, orient='index')

        bt_df.loc[:, 'option_pnl'] = bt_df['pv'].diff()

        bt_df.loc[:, 'delta_st1_pnl'] = bt_df['st1'].diff() * bt_df['delta_st1'].shift()
        bt_df.loc[:, 'delta_st2_pnl'] = bt_df['st2'].diff() * bt_df['delta_st2'].shift()
        bt_df.loc[:, 'delta_pnl'] = bt_df['delta_st1_pnl'] + bt_df['delta_st2_pnl']

        # bt_df.loc[:, 'gamma_st1_pnl'] = bt_df['st1'].diff().pow(2) * bt_df['gamma_st1'].shift() * 0.5
        # bt_df.loc[:, 'gamma_st2_pnl'] = bt_df['st2'].diff().pow(2) * bt_df['gamma_st2'].shift() * 0.5
        # bt_df.loc[:, 'gamma_pnl'] = bt_df['gamma_st1_pnl'] + bt_df['gamma_st2_pnl']
        #
        # bt_df.loc[:, 'x_gamma_pnl'] = bt_df['st1'].diff() * bt_df['st2'].diff() * bt_df['x_gamma'].shift()
        #
        # bt_df.loc[:,'explained_pnl'] = bt_df['delta_pnl'] + bt_df['gamma_pnl'] + bt_df['x_gamma_pnl'] + bt_df['theta']

        bt_df.loc[:, 'delta_cumpnl'] = bt_df['delta_pnl'].cumsum()
        bt_df.loc[:, 'option_cumpnl'] = bt_df['option_pnl'].cumsum()

        return bt_df

