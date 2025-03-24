import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from glob import glob
from scipy.stats import gaussian_kde

def load_backtest_data(data_dir='data'):
    """
    Load all backtest data files from the specified directory
    Returns a dictionary of DataFrames with historical data for each ticker
    """
    # Get the most recent timestamp from the data files
    files = glob(os.path.join(data_dir, 'backtest_data_*.csv'))
    if not files:
        raise FileNotFoundError("No backtest data files found in the data directory")
    
    # Extract timestamp from the first file
    timestamp = files[0].split('_')[-1].replace('.csv', '')
    
    # Load all files with this timestamp
    historical_data = {}
    for file in files:
        if timestamp in file:
            ticker = file.split('_')[2]
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            historical_data[ticker] = df
    
    return historical_data

def calculate_returns(df):
    """Calculate daily returns from price data"""
    df['returns'] = df['close'].pct_change()
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators including factor-based signals"""
    # Moving averages (adjusted for hourly data)
    df['SMA_8'] = df['close'].rolling(window=8).mean()  # 1 trading day
    df['SMA_40'] = df['close'].rolling(window=40).mean()  # 1 trading week
    
    # RSI (adjusted for hourly data)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=8).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=8).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (adjusted for hourly data)
    exp1 = df['close'].ewm(span=6, adjust=False).mean()
    exp2 = df['close'].ewm(span=12, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=4, adjust=False).mean()
    
    # Factor-based signals
    # 1. Momentum (shorter timeframe)
    df['momentum'] = df['returns'].rolling(window=4).mean()
    
    # 2. Value (price relative to moving average)
    df['value'] = df['close'] / df['SMA_40'] - 1
    
    # 3. Volatility (shorter timeframe)
    df['volatility'] = df['returns'].rolling(window=8).std()
    
    # 4. Volume trend (shorter timeframe)
    df['volume_ma'] = df['volume'].rolling(window=8).mean()
    df['volume_trend'] = df['volume'] / df['volume_ma'] - 1
    
    return df

def winsorize_returns(returns, lower=-0.1, upper=0.1):
    """Winsorize returns to handle outliers"""
    return returns.clip(lower=lower, upper=upper)

def calculate_position_size(volatility, signal_strength, max_position_size=0.25):
    """
    Calculate position size based on volatility and signal strength
    """
    # Base position size inversely proportional to volatility
    base_size = max_position_size / (1 + volatility)
    
    # Scale by signal strength (2 signals = 50%, 3 signals = 75%, 4 signals = 100%)
    signal_scale = min(signal_strength / 4, 1.0)
    
    return base_size * signal_scale

def backtest_strategy(df, initial_capital=100000):
    """
    Backtest the trading strategy
    """
    df = df.copy()
    
    # Calculate signals
    df['position'] = 0.0
    
    # Factor-based signals (adjusted for hourly data)
    df['momentum_signal'] = df['returns'].rolling(window=12).mean() > -0.0005
    df['value_signal'] = df['returns'].rolling(window=24).std() < 0.02
    df['RSI_signal'] = (df['RSI'] < 70) & (df['RSI'] > 30)
    df['MACD_signal'] = df['MACD'] > df['Signal_Line'] - 0.0005
    
    # Count positive signals
    df['signal_strength'] = (
        df['momentum_signal'].astype(int) +
        df['value_signal'].astype(int) +
        df['RSI_signal'].astype(int) +
        df['MACD_signal'].astype(int)
    )
    
    # Print signal analysis
    print("\nSignal Analysis:")
    print(f"Momentum signals: {df['momentum_signal'].sum()}")
    print(f"Value signals: {df['value_signal'].sum()}")
    print(f"RSI signals: {df['RSI_signal'].sum()}")
    print(f"MACD signals: {df['MACD_signal'].sum()}")
    print(f"Average signal strength: {df['signal_strength'].mean():.2f}")
    print(f"Number of strong signals (>=2): {(df['signal_strength'] >= 2).sum()}")
    
    # Calculate position size based on volatility and signal strength
    df['position_size'] = df.apply(lambda x: calculate_position_size(x['volatility'], x['signal_strength']), axis=1)
    
    # Create mask for positions
    mask = df['signal_strength'] >= 2
    
    # Assign positions using the mask
    df.loc[mask, 'position'] = df.loc[mask, 'position_size']
    
    # Exit positions when less than 2 signals are positive
    df.loc[~mask, 'position'] = 0
    
    # Implement stop-loss and take-profit
    stop_loss = -0.02  # 2% stop loss
    take_profit = 0.05  # 5% take profit
    
    # Track cumulative returns for each position
    df['cumulative_returns'] = df['returns'].cumsum()
    
    # Exit positions at stop-loss or take-profit
    df.loc[df['cumulative_returns'] <= stop_loss, 'position'] = 0
    df.loc[df['cumulative_returns'] >= take_profit, 'position'] = 0
    
    # Reset cumulative returns when position is closed
    df.loc[df['position'] == 0, 'cumulative_returns'] = 0
    
    # Debug print positions
    print(f"Number of positions taken: {(df['position'] > 0).sum()}")
    print(f"Average position size: {df['position'].mean():.4f}")
    
    # Winsorize returns to handle outliers
    df['returns'] = winsorize_returns(df['returns'], lower=-0.05, upper=0.05)  # Tighter for hourly
    
    # Calculate portfolio value
    df['portfolio_value'] = initial_capital
    df['position_returns'] = df['position'] * df['returns']
    df['portfolio_value'] = initial_capital * (1 + df['position_returns']).cumprod()
    
    return df

def plot_backtest_results(df, ticker):
    """Plot backtest results with additional metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot price and indicators
    ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
    ax1.plot(df.index, df['SMA_8'], label='8-day SMA', alpha=0.7)
    ax1.plot(df.index, df['SMA_40'], label='40-day SMA', alpha=0.7)
    ax1.set_title(f'{ticker} Price and Indicators')
    ax1.legend()
    
    # Plot technical indicators
    ax2.plot(df.index, df['RSI'], label='RSI', alpha=0.7)
    ax2.plot(df.index, df['MACD'], label='MACD', alpha=0.7)
    ax2.plot(df.index, df['Signal_Line'], label='Signal Line', alpha=0.7)
    ax2.set_title('Technical Indicators')
    ax2.legend()
    
    # Plot portfolio value and position size
    ax3.plot(df.index, df['portfolio_value'], label='Portfolio Value', alpha=0.7)
    ax3.plot(df.index, df['position'] * 100000, label='Position Size', alpha=0.7)
    ax3.set_title('Portfolio Value and Position Size')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(results):
    """Calculate performance metrics with proper handling of edge cases"""
    total_return = (results['portfolio_value'].iloc[-1] / 100000 - 1) * 100
    
    # Only calculate Sharpe ratio if we have trades
    if results['position_returns'].std() > 0:
        sharpe_ratio = np.sqrt(252) * results['position_returns'].mean() / results['position_returns'].std()
    else:
        sharpe_ratio = 0
    
    max_drawdown = (results['portfolio_value'] / results['portfolio_value'].cummax() - 1).min() * 100
    avg_position_size = results['position'].mean() * 100
    
    # Calculate win rate only for periods with positions
    trades = results[results['position'] > 0]
    if len(trades) > 0:
        win_rate = (trades['position_returns'] > 0).mean() * 100
    else:
        win_rate = 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_position_size': avg_position_size,
        'win_rate': win_rate
    }

def run_backtest():
    """Main function to run the backtest"""
    # Load data
    print("Loading backtest data...")
    historical_data = load_backtest_data()
    
    results_dict = {}
    
    # Run backtest for each ticker
    for ticker, df in historical_data.items():
        print(f"\nRunning backtest for {ticker}...")
        
        # Calculate returns and indicators
        df = calculate_returns(df)
        df = calculate_technical_indicators(df)
        
        # Run strategy
        results = backtest_strategy(df)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(results)
        
        print(f"\nPerformance Metrics for {ticker}:")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Average Position Size: {metrics['avg_position_size']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        
        results_dict[ticker] = {
            'results': results,
            'metrics': metrics
        }
    
    return results_dict

if __name__ == "__main__":
    run_backtest() 