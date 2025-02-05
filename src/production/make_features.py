import ta
import pandas as pd

def calculate_incremental_quadratic_variation(df, price_column='close', time_step=1):
    prices = df[price_column].values
    differences = np.diff(prices)
    squared_differences = np.square(differences)
    cumulative_qv = np.cumsum(squared_differences) / time_step
    cumulative_qv = np.pad(cumulative_qv, (1, 0), 'constant')
    incremental_qv = np.diff(cumulative_qv, prepend=0)
    incremental_qv_series = pd.Series(incremental_qv, index=df.index, name='incremental_qv')
    return incremental_qv_series

def create_target(df, close_col, num_candles):
    """
    Create a TARGET column based on the highest and lowest close prices within a range of num_candles.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        close_col (str): The name of the close column.
        num_candles (int): The number of candles either side to check for highest and lowest prices.

    Returns:
        pd.DataFrame: The DataFrame with the new TARGET column.
    """
    close_prices = df[close_col].values
    n = len(close_prices)
    target = np.zeros(n)

    for i in range(n):
        start = max(0, i - num_candles)
        end = min(n, i + num_candles + 1)
        local_window = close_prices[start:end]

        if close_prices[i] == max(local_window) and local_window.tolist().count(close_prices[i]) == 1:
            target[i] = 1
        elif close_prices[i] == min(local_window) and local_window.tolist().count(close_prices[i]) == 1:
            target[i] = 2

    df['TARGET'] = target
    return df

def make_features(df):
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
        df.drop(columns=['datetime'], inplace=True)
        #quadratic variation
        df['qv'] = calculate_incremental_quadratic_variation(df,'close')
        df['qv_ema3'] = df['qv'].ewm(span=3, adjust=False).mean()
        df['qv_ema5'] = df['qv'].ewm(span=5, adjust=False).mean()
        df['qv_ema10'] = df['qv'].ewm(span=10, adjust=False).mean()
        df['qv_ema30'] = df['qv'].ewm(span=30, adjust=False).mean()

        df = df[['open', 'high', 'low', 'close', 'tick_volume','hour','day_of_week','qv','qv_ema3','qv_ema5','qv_ema10','qv_ema30']]
        # Get target
        df = create_target(df,'close',15)
        # EMA
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=30, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=40, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        # RSI
        df['rsi_3'] = ta.momentum.RSIIndicator(df['close'], window=3).rsi()
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        df['rsi_28'] = ta.momentum.RSIIndicator(df['close'], window=28).rsi()
        # MACD
        df['macd'] = ta.trend.MACD(df['close']).macd()
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_hband'] = bollinger.bollinger_hband()
        df['bollinger_lband'] = bollinger.bollinger_lband()

        # Average True Range (ATR)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        # Stochastic Oscillator
        stochastic = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stochastic.stoch()
        df['stoch_d'] = stochastic.stoch_signal()

        # Rate of Change (ROC)
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()

        # Commodity Channel Index (CCI)
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()

        # On-Balance Volume (OBV)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['tick_volume']).on_balance_volume()

        return df