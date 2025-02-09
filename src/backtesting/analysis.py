import pandas as pd
import numpy as np

# take results_df

# PRINT

# returns over time plotted
# number of wins
# number of losses
# sharpe ratio
# alpha


# picture.to_png('/src/data/{env_variables}/graph1')
# variables.to_json('/src/data/{env_variables}/variables')

# merge results_df to full backtesting data dataframe
# write function that adds column that indicates trade outcome
# add fee column (fixed everytime we trade)

# summarisation:
# number of wins, number of losses

# baso same thing but plotted over time
# 

def analysis(full_backtesting_df, results_df):
    print(full_backtesting_df)
    print(results_df)

    analysis_df = full_backtesting_df.join(results_df, how='left')  
    analysis_df2 = calculate_realised_profit(analysis_df)

    print(analysis_df2)


import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def calculate_realised_profit(df):
    """
    Adds two columns to the DataFrame:
      - 'realised_profit': The profit/loss for a trade based on future price data.
      - 'exit_index': The datetime index where the trade's take profit or stop loss was hit.
    
    Assumptions:
      - The DataFrame index is datetime.
      - 'trade': 0 = no trade, 1 = long, 2 = short.
      - Trade entry is assumed at the 'close' price on the signal row.
      - For long trades, the assumed intrabar price sequence is: open -> high -> low.
      - For short trades, the assumed sequence is: open -> low -> high.
    """
    n = len(df)
    
    # Convert necessary columns to NumPy arrays for speed.
    trade     = df['trade'].values
    stop      = df['stop_loss'].values
    tp        = df['take_profit'].values
    close_p   = df['Close'].values
    open_p    = df['Open'].values
    high_p    = df['High'].values
    low_p     = df['Low'].values
    index_arr = df.index.values  # This should be a numpy array of np.datetime64[ns]
    
    # Create arrays for the output.
    realised_profit = np.full(n, np.nan, dtype=float)
    # Specify the datetime dtype explicitly to avoid type-casting issues.
    exit_index = np.full(n, np.datetime64('NaT', 'ns'), dtype='datetime64[ns]')
    
    # Loop through each row.
    for i in range(n):
        if trade[i] == 0:
            continue  # No trade on this row.
            
        entry_price = close_p[i]
        trade_stop  = stop[i]
        trade_tp    = tp[i]
        
        # Long trade: look for an upward exit.
        if trade[i] == 2:
            for j in range(i + 1, n):
                op = open_p[j]
                hi = high_p[j]
                lo = low_p[j]
                # Check conditions in a defined order:
                if op >= trade_tp:
                    realised_profit[i] = trade_tp - entry_price
                    exit_index[i]      = index_arr[j]
                    break
                elif op <= trade_stop:
                    realised_profit[i] = trade_stop - entry_price
                    exit_index[i]      = index_arr[j]
                    break
                elif hi >= trade_tp:
                    realised_profit[i] = trade_tp - entry_price
                    exit_index[i]      = index_arr[j]
                    break
                elif lo <= trade_stop:
                    realised_profit[i] = trade_stop - entry_price
                    exit_index[i]      = index_arr[j]
                    break
                    
        # Short trade: look for a downward exit.
        elif trade[i] == 1:
            for j in range(i + 1, n):
                op = open_p[j]
                hi = high_p[j]
                lo = low_p[j]
                if op <= trade_tp:
                    realised_profit[i] = entry_price - trade_tp
                    exit_index[i]      = index_arr[j]
                    break
                elif op >= trade_stop:
                    realised_profit[i] = entry_price - trade_stop
                    exit_index[i]      = index_arr[j]
                    break
                elif lo <= trade_tp:
                    realised_profit[i] = entry_price - trade_tp
                    exit_index[i]      = index_arr[j]
                    break
                elif hi >= trade_stop:
                    realised_profit[i] = entry_price - trade_stop
                    exit_index[i]      = index_arr[j]
                    break
    
    # Add the computed arrays as new columns.
    df['realised_profit'] = realised_profit
    df['exit_time'] = exit_index
    return df





