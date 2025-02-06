from production.make_features import make_features
from production.models.ATR import ATR
from production.models.classifier import classifier
from backtesting.get_backtesting_data import get_backtesting_data
import pandas as pd

# TODO: replace with variables straight from .env 
training_period_data_size = 1000
# format YYYY-MM-DD HH:MM
backtest_start_date_time = "2024.05.26 18:00"
# format YYYY-MM-DD HH:MM
backtest_end_date_time = "2024.05.26 20:00"

backtest_start_date_time = pd.to_datetime(backtest_start_date_time)
backtest_end_date_time = pd.to_datetime(backtest_end_date_time)

# TODO: adapt this to initiate correct columns
results_df = pd.DataFrame()

full_backtesting_data = get_backtesting_data().iloc[::-1].reset_index(drop=True)

# Get first start point and final start point
first_training_end_index = full_backtesting_data.index[full_backtesting_data['Time'] == backtest_start_date_time].tolist()[0]
final_training_end_index = full_backtesting_data.index[full_backtesting_data['Time'] == backtest_end_date_time].tolist()[0]

for training_end_point in range(first_training_end_index, final_training_end_index + 1):

    training_start_point = training_end_point - training_period_data_size + 1
    training_df = full_backtesting_data.iloc[training_start_point:training_end_point + 1]

    print(training_df)

    features_df = make_features(training_df)

    print(features_df)
    

# for loop through dates and times

    # data = (date and time) < data < (date and time)

    # call make features

    # call train classifier (input: data) (output: buy, sell, no trade)

    # append classifier outcome to results_df

    # if not no trade:

        # call ATR model (input: data) (output: take profit, stop loss)

        # append ATR outcome to results_df


# call get analysis (input: results_df) (output: plots, summarisation etc)


