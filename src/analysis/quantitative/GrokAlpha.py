import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

# Sample DataFrame setup (replace with your real data)
# df = pd.DataFrame({'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...]})
# For demonstration, assuming df exists with 1-minute SPY data

# 1. Momentum Burst Signal
def momentum_burst_signal(df, ema_period=5, price_threshold=0.001, volume_multiplier=2):
    df['ema'] = EMAIndicator(df['close'], window=ema_period).ema_indicator()
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['price_change'] = df['close'].pct_change()
    
    df['signal'] = np.where(
        (df['close'] > df['ema']) & 
        (df['price_change'] > price_threshold) & 
        (df['volume'] > df['avg_volume'] * volume_multiplier), 
        'buy_call', 
        np.where(
            (df['close'] < df['ema']) & 
            (df['price_change'] < -price_threshold) & 
            (df['volume'] > df['avg_volume'] * volume_multiplier), 
            'buy_put', 
            None
        )
    )
    return df

# 2. Reversal at Key Levels (using VWAP and candlestick rejection)
def reversal_signal(df):
    df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).vwap()
    
    # Simple hammer/doji detection (adjust thresholds as needed)
    df['body'] = abs(df['close'] - df['open'])
    df['lower_wick'] = df['low'].where(df['close'] > df['open'], df['close']) - df['open']
    df['is_hammer'] = (df['lower_wick'] > 2 * df['body']) & (df['close'] > df['vwap'])
    df['is_doji'] = (df['body'] < 0.0001 * df['close']) & (df['close'] < df['vwap'])
    
    df['signal'] = np.where(df['is_hammer'], 'buy_call', 
                           np.where(df['is_doji'], 'buy_put', None))
    return df

# 3. RSI Divergence
def rsi_divergence_signal(df, rsi_period=5):
    df['rsi'] = RSIIndicator(df['close'], window=rsi_period).rsi()
    df['price_diff'] = df['close'].diff()
    df['rsi_diff'] = df['rsi'].diff()
    
    # Bullish divergence: price higher low, RSI lower low
    df['bullish_div'] = (df['price_diff'] > 0) & (df['rsi_diff'] < 0) & (df['rsi'] < 30)
    # Bearish divergence: price lower high, RSI higher high
    df['bearish_div'] = (df['price_diff'] < 0) & (df['rsi_diff'] > 0) & (df['rsi'] > 70)
    
    df['signal'] = np.where(df['bullish_div'], 'buy_call', 
                           np.where(df['bearish_div'], 'buy_put', None))
    return df

# 4. Volume Breakout
def volume_breakout_signal(df, lookback=5, volume_multiplier=3):
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['high_break'] = df['high'].rolling(window=lookback).max().shift(1)
    df['low_break'] = df['low'].rolling(window=lookback).min().shift(1)
    
    df['signal'] = np.where(
        (df['close'] > df['high_break']) & 
        (df['volume'] > df['avg_volume'] * volume_multiplier), 
        'buy_call', 
        np.where(
            (df['close'] < df['low_break']) & 
            (df['volume'] > df['avg_volume'] * volume_multiplier), 
            'buy_put', 
            None
        )
    )
    return df

# 5. MACD Zero-Line Cross
def macd_zero_cross_signal(df, fast=3, slow=10, signal=5, ema_period=8):
    macd = MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
    df['macd'] = macd.macd()
    df['ema'] = EMAIndicator(df['close'], window=ema_period).ema_indicator()
    
    df['macd_cross_up'] = (df['macd'].shift(1) < 0) & (df['macd'] > 0) & (df['close'] > df['ema'])
    df['macd_cross_down'] = (df['macd'].shift(1) > 0) & (df['macd'] < 0) & (df['close'] < df['ema'])
    
    df['signal'] = np.where(df['macd_cross_up'], 'buy_call', 
                           np.where(df['macd_cross_down'], 'buy_put', None))
    return df

# 6. Opening Range Breakout (ORB)
def orb_signal(df, orb_period=15):
    # Assuming df index is datetime, get first 'orb_period' minutes
    df['time'] = df.index.time
    orb_high = df['high'][:orb_period].max()
    orb_low = df['low'][:orb_period].min()
    
    df['signal'] = np.where(
        (df['close'] > orb_high) & (df['volume'] > df['volume'].shift(1)), 
        'buy_call', 
        np.where(
            (df['close'] < orb_low) & (df['volume'] > df['volume'].shift(1)), 
            'buy_put', 
            None
        )
    )
    return df

# Example usage (replace 'df' with your actual DataFrame)
# df = momentum_burst_signal(df)
# df = reversal_signal(df)
# df = rsi_divergence_signal(df)
# df = volume_breakout_signal(df)
# df = macd_zero_cross_signal(df)
# df = orb_signal(df)

# Print signals
# print(df[['close', 'volume', 'signal']].tail())