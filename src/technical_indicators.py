import pandas as pd
import numpy as np
import talib

def calculate_sma(df, window):

    '''
    Function to calculate Simple Moving Average (SMA)
    '''

    data = df.copy()

    data[f'TI_SMA_{window}_Feature'] = data['close'].rolling(window=window).mean()

    return data

def calculate_ema(df, window):
    
    '''
    Function to calculate Exponential Moving Average (EMA)
    '''

    data = df.copy()

    data[f'TI_EMA_{window}_Feature'] = data['close'].ewm(span=window, adjust=False).mean()

    return data

def calculate_rsi(df, window):
    '''
    Function to calculate Relative Strength Index (RSI)
    '''

    data = df.copy()

    delta = data['close'].diff()
    loss = (delta.where(delta < 0, 0))
    gain = (-delta.where(delta > 0, 0))
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = abs(avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    data[f'MI_RSI_{window}_Feature'] = rsi

    return data

def calculate_macd(df, short_window, long_window, signal_window):
    '''
    Function to calculate Moving Average Convergence Divergence (MACD)
    '''
    
    data = df.copy()

    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()
    data[f'MI_MACD_{short_window}_{long_window}_Feature'] = short_ema - long_ema
    data[f'MI_Signal_Line_{short_window}_{long_window}_Feature'] = data[f'MI_MACD_{short_window}_{long_window}_Feature'].ewm(span=signal_window, adjust=False).mean()

    return data

def calculate_bollinger_bands(df, window, num_of_std):
    """
    Bollinger Bands (BB)
    """
    data = df.copy()
    data[f'VolI_BB_Middle_Band_{window}_{num_of_std}_Feature'] = data['close'].rolling(window=window).mean()
    data[f'VolI_BB_Upper_Band_{window}_{num_of_std}_Feature'] = data[f'VolI_BB_Middle_Band_{window}_{num_of_std}_Feature'] + (data['close'].rolling(window=window).std() * num_of_std)
    data[f'VolI_BB_Lower_Band_{window}_{num_of_std}_Feature'] = data[f'VolI_BB_Middle_Band_{window}_{num_of_std}_Feature'] - (data['close'].rolling(window=window).std() * num_of_std)

    return data

def calculate_stochastic_oscillator(df, window):

    data = df.copy()

    data[f'MI_Stochastic_Osclilator_percK_{window}_Feature'] = ((data['close'] - data['low'].rolling(window=window).min()) / (data['high'].rolling(window=window).max() - data['low'].rolling(window=window).min())) * 100
    data[f'MI_Stochastic_Osclilator_percD_{window}_Feature'] = data[f'MI_Stochastic_Osclilator_percK_{window}_Feature'].rolling(window=3).mean()

    return data

def calculate_atr(df, window):
    """
    Average True Range (ATR)
    """
    data = df.copy()
    data[f'VolI_ATR_{window}_Feature'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=window)

    return data

def compute_technical_indicators(df):
    '''
    Function to compute all technical indicators
    '''

    data = df.copy()

    data = calculate_sma(data, 5)
    data = calculate_sma(data, 14)
    data = calculate_sma(data, 21)

    data = calculate_ema(data, 5)
    data = calculate_ema(data, 14)
    data = calculate_ema(data, 21)

    data = calculate_rsi(data, 5)
    data = calculate_rsi(data, 14)
    data = calculate_rsi(data, 21)

    
    data = calculate_macd(data, 12, 26, 9)
    data = calculate_macd(data, 8, 17, 9)
    data = calculate_macd(data, 13, 30, 10)

    data = calculate_bollinger_bands(data, 20, 1)
    data = calculate_bollinger_bands(data, 20, 2)
    data = calculate_bollinger_bands(data, 20, 3)

    data = calculate_stochastic_oscillator(data, 5)
    data = calculate_stochastic_oscillator(data, 14)
    data = calculate_stochastic_oscillator(data, 21)

    data = calculate_atr(data, 5)
    data = calculate_atr(data, 14)
    data = calculate_atr(data, 21)

    return data

def OHLC_features(df):
    '''
    Function to compute OHLC features
    '''

    data = df.copy()

    # log returns
    data['close_log_returns'] = np.log(data['close'] / data['close'].shift(1))
    data['open_log_returns'] = np.log(data['open'] / data['open'].shift(1))
    data['high_log_returns'] = np.log(data['high'] / data['high'].shift(1))
    data['low_log_returns'] = np.log(data['low'] / data['low'].shift(1))

    # percentage change
    data['close_perc_change'] = data['close'].pct_change()
    data['open_perc_change'] = data['open'].pct_change()
    data['high_perc_change'] = data['high'].pct_change()
    data['low_perc_change'] = data['low'].pct_change()

    # high low range
    data['high_low_range'] = data['high'] - data['low']

    # close open range
    data['close_open_range'] = data['close'] - data['open']
    
    return data