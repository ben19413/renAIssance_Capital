from production.make_features import make_features
from production.models.ATR import ATR
from production.models.classifier import classifier
from backtesting.get_backtesting_data import get_backtesting_data
import pandas as pd
from datetime import timedelta
import numpy as np

# TODO: replace with variables straight from .env
training_period_data_size = 1000
# format YYYY-MM-DD HH:MM
backtest_start_date_time = "2024.05.26 20:00"
# format YYYY-MM-DD HH:MM
backtest_end_date_time = "2024.05.26 21:00"

backtest_start_date_time = pd.to_datetime(backtest_start_date_time)
backtest_end_date_time = pd.to_datetime(backtest_end_date_time)

# TODO: adapt this to initiate correct columns
results_df = pd.DataFrame(columns=["trade", "stop_loss", "take_profit"])

full_backtesting_data = get_backtesting_data().iloc[::-1]

filtered_backtesting_data = full_backtesting_data.loc[
    backtest_start_date_time:backtest_end_date_time
]
for training_end_point in filtered_backtesting_data.index:

    training_start_point = training_end_point - timedelta(
        hours=training_period_data_size
    )

    training_df = full_backtesting_data.loc[training_start_point:training_end_point]

    features_df = make_features(training_df)

    trade = classifier(features_df)

    if trade != 0:
        stop_loss, take_profit = ATR(features_df, trade)
    else:
        stop_loss = None
        take_profit = None

    iteration_df = pd.DataFrame(
        {
            "trade": trade,
            "stop_loss": [stop_loss if stop_loss is not None else np.nan],
            "take_profit": [take_profit if take_profit is not None else np.nan],
        }
    )

    results_df = pd.concat([results_df, iteration_df], ignore_index=True)

results_df.index = filtered_backtesting_data.index

# call get analysis (input: results_df) (output: plots, summarisation etc)
