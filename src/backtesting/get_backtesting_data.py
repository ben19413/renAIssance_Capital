import pandas as pd

# use metatrader api to get data set between backtest_end_date and (backtest_start_date - training_period_length)

# TODO: REPLACE HACK WITH LIVE DATA SOURCE


def get_backtesting_data():

    full_backtesting_data = pd.read_csv("../files/EURUSD_full_test.csv")
    full_backtesting_data.index = pd.to_datetime(full_backtesting_data["Time"])
    return full_backtesting_data
