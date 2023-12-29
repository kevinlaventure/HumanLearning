import numpy as np
import pandas as pd
from module.pricing import black_scholes_option

def black_scholes_delta_hedged(close_price_dict, option_type, strike, maturity, roll_day,  sigma, r=0.0, periods_per_year=252):
    """Compute Option Strategy Backtest

    Args:
        close_price_dict (dict): close price timeseries
        option_type (str): 'call' for a call option, 'put' for a put option.
        strike (float): strike in percentage of the close price
        maturity (int): number of business day to expiry
        roll_day (int): number of business day to roll
        sigma (float): volatility of the underlying stock
        r (float): risk-free rate.
        periods_per_year (int, optional): number of business day per year, defaults to 252.

    Returns:
        dict: market data, greeks and backtest P&L
    """
    
    bt_dict = dict()
    trade_id = 0
    previous_business_day = None
    business_day_to_expiry = maturity

    def fill_bt(bt_dict, current_business_day, trade_id, rebalancing, business_day_to_expiry, close_price, K, sigma, periods_per_year):

        bt_dict[(current_business_day, trade_id, 'Rebalancing')] = rebalancing
        bt_dict[(current_business_day, trade_id, 'BusinessDayToExpiry')] = business_day_to_expiry

        bt_dict[(current_business_day, trade_id, 'S')] = close_price
        bt_dict[(current_business_day, trade_id, 'K')] = K
        bt_dict[(current_business_day, trade_id, 'Sigma')] = sigma
        bt_dict[(current_business_day, trade_id, 'T')] = business_day_to_expiry / periods_per_year
        
        pricing_results = black_scholes_option(
            option_type=option_type, 
            S=bt_dict[(current_business_day, trade_id, 'S')], 
            K=bt_dict[(current_business_day, trade_id, 'K')], 
            T=bt_dict[(current_business_day, trade_id, 'T')], 
            sigma=sigma,
            r=r)
        
        bt_dict[(current_business_day, trade_id, 'Units')]  = 1
        
        for key in pricing_results:
            bt_dict[(current_business_day, trade_id, key)] = pricing_results[key]

        bt_dict[(current_business_day, trade_id, 'GammaCash')] =  (bt_dict[(current_business_day, trade_id, 'Gamma')] * close_price**2) / 100

        return bt_dict
    
    for current_business_day, close_price in close_price_dict.items():
        
        if previous_business_day is None:
            rebalancing = True
            K = close_price * strike
            bt_dict = fill_bt(bt_dict, current_business_day, trade_id, rebalancing, business_day_to_expiry, close_price, K, sigma, periods_per_year)
            business_day_to_expiry -= 1
            previous_business_day = current_business_day
        else:
            rebalancing = False
            K = bt_dict[(previous_business_day, trade_id, 'K')]
            bt_dict = fill_bt(bt_dict, current_business_day, trade_id, rebalancing, business_day_to_expiry, close_price, K, sigma, periods_per_year)
                
            if business_day_to_expiry == roll_day:
                trade_id += 1
                rebalancing = True
                K = close_price * strike
                business_day_to_expiry = maturity
                bt_dict = fill_bt(bt_dict, current_business_day, trade_id, rebalancing, business_day_to_expiry, close_price, K, sigma, periods_per_year)
                
            business_day_to_expiry -= 1
            previous_business_day = current_business_day

    bt_df = pd.Series(bt_dict).unstack()
    bt_df.loc[:, 'GammaPnL'] = (bt_df['GammaCash'].shift() * 50 * bt_df['S'].pct_change().pow(2)) * bt_df['Units'].shift()
    bt_df.loc[:, 'ThetaPnL'] = bt_df['Theta'].shift() * bt_df['Units'].shift()
    bt_df.loc[:, 'GammaThetaPnL'] = bt_df['GammaPnL'] + bt_df['ThetaPnL']
    bt_df.loc[:, 'OptionPnL'] = bt_df['PV'].diff() * bt_df['Units'].shift()
    bt_df.loc[:, 'DeltaPnL'] = bt_df['S'].diff() * bt_df['Delta'].shift() * bt_df['Units'].shift()
    bt_df.loc[:, 'OptionHedgedPnL'] = bt_df['OptionPnL'] - bt_df['DeltaPnL']
    bt_df.loc[bt_df['Rebalancing']==True, ['GammaPnL', 'ThetaPnL', 'GammaThetaPnL', 'OptionPnL', 'DeltaPnL', 'OptionHedgedPnL']] = 0
    bt_df.loc[:, 'CumOptionHedgedPnL'] = bt_df['OptionHedgedPnL'].cumsum()
    bt_df.index = bt_df.index.set_names(['Date', 'TradeId'])
    bt_df = bt_df.reset_index().set_index('Date')
    
    return bt_df
