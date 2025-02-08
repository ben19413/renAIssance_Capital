# retrieve price data via:
# call process_api_data (input: data sent via api trigger) (output: data in same format as data after filtering in backtesting)

# make features
# call train classifier (input: data) (output: buy, sell, no trade)
# if not no trade:
#       call train ATR (input: data) (output: take profit, stop loss)
# else:
#       give stop loss and take profit arbitary values
# Return prediction, stop loss, take profit
