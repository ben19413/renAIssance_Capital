import ta
import pandas as pd
import numpy as np

def calculate_incremental_quadratic_variation(df, price_column='Close', time_step=1):
    prices = df[price_column].values
    differences = np.diff(prices)
    squared_differences = np.square(differences)
    cumulative_qv = np.cumsum(squared_differences) / time_step
    cumulative_qv = np.pad(cumulative_qv, (1, 0), 'constant')
    incremental_qv = np.diff(cumulative_qv, prepend=0)
    incremental_qv_series = pd.Series(incremental_qv, index=df.index, name='incremental_qv')
    return incremental_qv_series

def create_target(df, Close_col, num_candles):
    """
    Create a TARGET column based on the highest and lowest Close prices within a range of num_candles.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        Close_col (str): The name of the Close column.
        num_candles (int): The number of candles either side to check for highest and lowest prices.

    Returns:
        pd.DataFrame: The DataFrame with the new TARGET column.
    """
    Close_prices = df[Close_col].values
    n = len(Close_prices)
    target = np.zeros(n)

    for i in range(n):
        start = max(0, i - num_candles)
        end = min(n, i + num_candles + 1)
        local_window = Close_prices[start:end]

        if Close_prices[i] == max(local_window) and local_window.tolist().count(Close_prices[i]) == 1:
            target[i] = 1
        elif Close_prices[i] == min(local_window) and local_window.tolist().count(Close_prices[i]) == 1:
            target[i] = 2

    df['Target'] = target
    return df

def make_features(df):
        df['Time'] = pd.to_datetime(df['Time'])
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek  # Monday=0, Sunday=6
        df.drop(columns=['Time'], inplace=True)
        #quadratic variation
        df['qv'] = calculate_incremental_quadratic_variation(df,'Close')
        df['qv_ema3'] = df['qv'].ewm(span=3, adjust=False).mean()
        df['qv_ema5'] = df['qv'].ewm(span=5, adjust=False).mean()
        df['qv_ema10'] = df['qv'].ewm(span=10, adjust=False).mean()
        df['qv_ema30'] = df['qv'].ewm(span=30, adjust=False).mean()
        # Get target
        df = create_target(df,'Close',15)
        # EMA
        df['ema5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['ema10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['ema30'] = df['Close'].ewm(span=30, adjust=False).mean()
        df['ema40'] = df['Close'].ewm(span=40, adjust=False).mean()
        df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
        # RSI
        df['rsi_3'] = ta.momentum.RSIIndicator(df['Close'], window=3).rsi()
        df['rsi_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['Close'], window=21).rsi()
        df['rsi_28'] = ta.momentum.RSIIndicator(df['Close'], window=28).rsi()
        # MACD
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()

        # Average True Range (ATR)
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()

        # Stochastic Oscillator
        stochastic = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['stoch_k'] = stochastic.stoch()
        df['stoch_d'] = stochastic.stoch_signal()

        # Rate of Change (ROC)
        df['roc'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()

        # Commodity Channel Index (CCI)
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()

        # On-Balance Volume (OBV)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        df = df.dropna() 

        return df