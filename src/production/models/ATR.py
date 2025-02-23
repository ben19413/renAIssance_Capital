import os


def ATR(features_df, trade, risk_to_reward):

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
