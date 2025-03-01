import os
import pandas as pd
import numpy as np


def ATR(features_df, predictions_df, risk_to_reward):

    boosted_predictions_df = pd.merge(predictions_df, features_df[['Close', 'ATR']], left_index=True, right_index=True)


    boosted_predictions_df["stop_loss"] = np.where(
        boosted_predictions_df["trade"] == 0, np.nan,
        np.where(
            boosted_predictions_df["trade"] == 2,
            boosted_predictions_df["Close"] - boosted_predictions_df["ATR"],
            boosted_predictions_df["Close"] + boosted_predictions_df["ATR"]
        )
    )

    boosted_predictions_df["take_profit"] = np.where(
        boosted_predictions_df["trade"] == 0, np.nan,
        np.where(
            boosted_predictions_df["trade"] == 2,
            boosted_predictions_df["Close"] + boosted_predictions_df["ATR"] * risk_to_reward,
            boosted_predictions_df["Close"] - boosted_predictions_df["ATR"] * risk_to_reward
        )
    )

    # Ensure ATR is NaN when trade == 0
    boosted_predictions_df["ATR"] = np.where(
        boosted_predictions_df["trade"] == 0, np.nan, boosted_predictions_df["ATR"]
    )

    boosted_predictions_df = boosted_predictions_df.drop(columns=["Close"])

    return boosted_predictions_df



def ATR_prod_refactor(features_df, trade, risk_to_reward):

    final_obs_df = features_df.tail(1)
    close_price = final_obs_df["Close"].iloc[0]
    ATR = final_obs_df["ATR"].iloc[0]

    if trade == 2:
        take_profit = close_price + risk_to_reward * ATR
        stop_loss = close_price - ATR
    else:
        take_profit = close_price - risk_to_reward * ATR
        stop_loss = close_price + ATR

    return float(stop_loss), float(take_profit), float(ATR)
