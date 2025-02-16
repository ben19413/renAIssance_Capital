import asyncio
import json
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from backtesting.get_backtesting_data import get_backtesting_data
from backtesting.analysis import analysis
from production.make_features import make_features
from production.models.ATR import ATR
from production.models.classifier import classifier

warnings.filterwarnings("ignore")

# Load configuration
CONFIG_PATH = os.getenv("config_path")
with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

backtest_start_date_time = pd.to_datetime(config["backtest_start_date_time"])
backtest_end_date_time = pd.to_datetime(config["backtest_end_date_time"])

full_backtesting_df = get_backtesting_data().iloc[::-1]
print("Full backtesting data shape:", full_backtesting_df.shape)
training_end_point_df = full_backtesting_df.loc[backtest_start_date_time:backtest_end_date_time]
print("Training endpoints count:", len(training_end_point_df))


async def process_training_point(training_end_point, full_backtesting_df, config):
    """
    Process one training endpoint:
      - Get the training window,
      - Generate features,
      - Get a trade signal and, if applicable,
      - Calculate stop loss and take profit.
    """
    loop = asyncio.get_running_loop()

    training_start_point = training_end_point - pd.Timedelta(hours=config["training_period_data_size"])
    training_df = full_backtesting_df.loc[training_start_point:training_end_point]

    # Offload synchronous work to a thread pool executor
    features_df = await loop.run_in_executor(None, make_features, training_df)
    trade = await loop.run_in_executor(None, classifier, features_df)

    if trade != 0:
        stop_loss, take_profit = await loop.run_in_executor(
            None, ATR, features_df, trade, config["risk_to_reward_ratio"]
        )
    else:
        stop_loss, take_profit = None, None

    iteration_df = pd.DataFrame({
        "trade": [trade],
        "stop_loss": [stop_loss if stop_loss is not None else np.nan],
        "take_profit": [take_profit if take_profit is not None else np.nan],
    })

    return training_end_point, iteration_df


async def main():
    # Get full backtesting data (synchronously)
    full_backtesting_df = get_backtesting_data().iloc[::-1]
    training_end_point_df = full_backtesting_df.loc[backtest_start_date_time:backtest_end_date_time]

    # Create a list of asynchronous tasks, one per training endpoint
    tasks = [
        process_training_point(training_end_point, full_backtesting_df, config)
        for training_end_point in training_end_point_df.index
    ]

    results = []
    # Use asyncio.as_completed to process tasks as they finish while showing progress
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await coro
        results.append(result)

    # Sort results by the training endpoint (to preserve the original order)
    results.sort(key=lambda x: x[0])
    iteration_dfs = [df for _, df in results]

    # Combine the iteration dataframes
    results_df = pd.concat(iteration_dfs, ignore_index=True)
    results_df.index = training_end_point_df.index

    # Run analysis (assumed to be synchronous)
    analysis(full_backtesting_df, results_df)

if __name__ == "__main__":
    asyncio.run(main())

