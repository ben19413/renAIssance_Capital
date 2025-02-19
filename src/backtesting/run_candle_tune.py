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
import json

warnings.filterwarnings("ignore")

CONFIG_PATH = os.getenv("config_path")

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)
for width in range(1, 31):
    backtest_start_date_time = pd.to_datetime(config["backtest_start_date_time"])
    backtest_end_date_time = pd.to_datetime(config["backtest_end_date_time"])

    results_df = pd.DataFrame(columns=["trade", "stop_loss", "take_profit" ,"ATR"])

    full_backtesting_df = get_backtesting_data().iloc[::-1]

    training_end_point_df = full_backtesting_df.loc[
        backtest_start_date_time:backtest_end_date_time
    ]
    for training_end_point in tqdm(training_end_point_df.index):

        training_start_point = training_end_point - timedelta(
            hours=config["training_period_data_size"]
        )

        training_df = full_backtesting_df.loc[training_start_point:training_end_point]

        features_df = make_features(training_df, width)

        trade = classifier(features_df)

        
        if trade != 0:
            stop_loss, take_profit, atr = ATR(features_df, trade, config["risk_to_reward_ratio"])
        else:
            stop_loss = None
            take_profit = None
            atr = None

        iteration_df = pd.DataFrame(
            {
                "trade": trade,
                "stop_loss": [stop_loss if stop_loss is not None else np.nan],
                "take_profit": [take_profit if take_profit is not None else np.nan],
                "ATR": [atr if atr is not None else np.nan],
            }
        )

        results_df = pd.concat([results_df, iteration_df], ignore_index=True)

    results_df.index = training_end_point_df.index
    print("Candle Width = " + str(width))
    analysis(full_backtesting_df, results_df, config, width)
