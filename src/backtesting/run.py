from backtesting.get_backtesting_data import get_backtesting_data
from production.models.classifier import classifier
from production.models.ATR import ATR
from production.make_features import make_features
from backtesting.analysis import trial_analysis, analysis

from tqdm import tqdm

import pandas as pd
from datetime import timedelta
import numpy as np
import os
import warnings
import json

warnings.filterwarnings("ignore")

CONFIG_PATH = os.getenv("config_path_development")

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)


config["general"]["backtest_start_date_time"] = pd.to_datetime(config["general"]["backtest_start_date_time"])
config["general"]["backtest_end_date_time"] = pd.to_datetime(config["general"]["backtest_end_date_time"])

# Initiate dict to store stats from each trial
stats_dict = {}
# Initiate dict to store predictions from each trial
predictions_dict = {}
# Initiate dataframe to store configs from each trial
config_df = pd.DataFrame(columns = ["instrument", "training_period_data_size", "target_width", "risk_to_reward_ratio"])

for trial, parameters in config["trials"].items():

    trial_config = config["trials"][trial]

    # Check if same classification has been run already
    trial_config_df = pd.DataFrame([trial_config])
    trial_config_df.insert(0, "Trial", trial)
    filter_columns = [col for col in trial_config if col != 'risk_to_reward_ratio']
    condition = True
    for col in filter_columns:
        condition &= config_df[col] == trial_config[col]
    repeat_df = config_df[condition]

    trial_config.update(config["general"]) 
    
    if repeat_df.empty:

        config_df = pd.concat([config_df, trial_config_df], ignore_index=True)

        predictions_df = pd.DataFrame(columns=["trade", "stop_loss", "take_profit", "ATR"])

        full_backtesting_df = get_backtesting_data(trial_config["instrument"])

        training_end_point_df = full_backtesting_df.loc[
            trial_config["backtest_start_date_time"]:trial_config["backtest_end_date_time"]
        ]
        for training_end_point in tqdm(training_end_point_df.index):
            training_start_point = training_end_point - timedelta(
                hours=trial_config["training_period_data_size"]
            )
            
            training_df = full_backtesting_df.loc[training_start_point:training_end_point]

            features_df = make_features(training_df, trial_config["target_width"])

            trade = classifier(features_df)

            if trade != 0:
                stop_loss, take_profit, atr = ATR(
                    features_df, trade, trial_config["risk_to_reward_ratio"]
                )
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

            predictions_df = pd.concat([predictions_df, iteration_df], ignore_index=True)

        predictions_df.index = training_end_point_df.index
    else:
        prediction_df = predictions_dict[repeat_df.head(1)['Trial'].iloc[0]]

    predictions_dict[trial] = predictions_df

    if predictions_df["trade"].sum() != 0:
        stats_dict = trial_analysis(full_backtesting_df, predictions_df, trial_config, stats_dict, trial)
    else:
        print("Analysis module off - no trades taken")

analysis(stats_dict)