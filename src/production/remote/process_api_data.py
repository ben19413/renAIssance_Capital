import pandas as pd

# process the data into the correct format (identical to output of get_backtesting_data)

# TODO: adjust to match data format received by MT5 bot 

def process_api_data(json):

    live_df = pd.DataFrame(json).iloc[::-1]
    live_df.index = pd.to_datetime(live_df["Time"])

    return live_df