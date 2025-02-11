from backtesting.get_backtesting_data import get_backtesting_data
from backtesting.analysis import analysis

from production.make_features import make_features
from production.models.ATR import ATR
from production.models.classifier import classifier
from tqdm import tqdm

import pandas as pd
from datetime import timedelta
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

backtest_start_date_time = pd.to_datetime(os.getenv("backtest_start_date_time"))
backtest_end_date_time = pd.to_datetime(os.getenv("backtest_end_date_time"))

results_df = pd.DataFrame(columns=["trade", "stop_loss", "take_profit"])

full_backtesting_df = get_backtesting_data().iloc[::-1]

training_end_point_df = full_backtesting_df.loc[
    backtest_start_date_time:backtest_end_date_time
]
for training_end_point in tqdm(training_end_point_df.index):

    training_start_point = training_end_point - timedelta(
        hours=int(os.getenv("training_period_data_size"))
    )

    training_df = full_backtesting_df.loc[training_start_point:training_end_point]

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

results_df.index = training_end_point_df.index

analysis(full_backtesting_df, results_df)
# call get analysis (input: results_df) (output: plots, summarisation etc)
