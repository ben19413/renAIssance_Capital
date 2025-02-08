import pandas as pd

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

    print(analysis_df)

