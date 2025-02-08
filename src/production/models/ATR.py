def ATR(features_df, trade):

    final_obs_df = features_df.tail(1)
    close_price = final_obs_df['Close'].iloc[0]
    ATR = final_obs_df['ATR'].iloc[0]

    if trade == 2:
        take_profit = close_price + ATR
        stop_loss = close_price - ATR
    else:
        take_profit = close_price - ATR
        stop_loss = close_price + ATR

    return stop_loss, take_profit

