import numpy as np
from ta.trend import EMAIndicator

def momentum_burst_signal(df, ema_period=5, price_threshold=0.001, volume_multiplier=2):
    # Calculate EMA and volume average
    df['ema'] = EMAIndicator(df['close'], window=ema_period).ema_indicator()
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['price_change'] = df['close'].pct_change()
    
    # Generate signals
    df['signal'] = np.where(
        (df['close'] > df['ema']) & 
        (df['price_change'] > price_threshold) & 
        (df['volume'] > df['avg_volume'] * volume_multiplier), 
        1,  # Buy call (long SPY)
        np.where(
            (df['close'] < df['ema']) & 
            (df['price_change'] < -price_threshold) & 
            (df['volume'] > df['avg_volume'] * volume_multiplier), 
            -1,  # Buy put (short SPY)
            0   # No signal
        )
    )
    return df

# Apply signal
df = momentum_burst_signal(df)