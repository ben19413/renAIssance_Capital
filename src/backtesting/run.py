from src.production import make_features
from src.production.models import ATR, classifier
import pandas as pd

# get full backtesting data via get_backtesting_data
results_df = pd.DataFrame()

# for loop through dates and times

    # data = (date and time) < data < (date and time)

    # call make features

    # call train classifier (input: data) (output: buy, sell, no trade)

    # append classifier outcome to results_df

    # if not no trade:

        # call train ATR (input: data) (output: stop loss)

        # append ATR outcome to results_df


# call get analysis (input: results_df) (output: plots, summarisation etc)


