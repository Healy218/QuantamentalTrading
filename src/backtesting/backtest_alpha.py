import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os
from glob import glob
from scipy.stats import gaussian_kde
from src.analysis.quantitative.alpha_factors import AlphaFactors, FactorConfig

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

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a given DataFrame
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with technical indicators
    """
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Calculate volatility
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    
    return df

def winsorize_returns(returns, lower=-0.1, upper=0.1):
    """Winsorize returns to handle outliers"""
    return returns.clip(lower=lower, upper=upper)

def calculate_position_size(volatility: float, signal_strength: int, max_position_size: float = 0.25) -> float:
    """
    Calculate position size based on volatility and signal strength
    
    Parameters
    ----------
    volatility : float
        Volatility of the asset
    signal_strength : int
        Number of positive signals (2-4)
    max_position_size : float
        Maximum position size as a fraction of portfolio
        
    Returns
    -------
    float
        Position size as a fraction of portfolio
    """
    # Base position size inversely proportional to volatility
    base_size = max_position_size * (1 / (1 + volatility))
    
    # Scale position size based on signal strength
    if signal_strength == 2:
        return base_size * 0.5
    elif signal_strength == 3:
        return base_size * 0.75
    elif signal_strength == 4:
        return base_size
    else:
        return 0.0

def backtest_strategy(ticker: str, data: pd.DataFrame, alpha_factors: AlphaFactors) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest the strategy for a given ticker
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    data : pd.DataFrame
        Historical data
    alpha_factors : AlphaFactors
        Alpha factors instance
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        DataFrame with positions and returns, and dictionary with metrics
    """
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Calculate alpha factors
    factors = alpha_factors.calculate_all_factors()
    
    # Combine factors with weights
    weights = {
        'momentum': 0.3,
        'mean_reversion': 0.2,
        'overnight_sentiment': 0.3,
        'volume_trend': 0.2
    }
    combined_alpha = alpha_factors.combine_factors(factors, weights)
    
    # Initialize portfolio tracking
    portfolio_value = 100000  # Starting with $100,000
    position = 0
    position_size = 0
    cumulative_return = 0
    trades = []
    portfolio_values = []
    
    # Risk management parameters
    stop_loss = -0.02  # 2% stop loss
    take_profit = 0.05  # 5% take profit
    
    # Track performance metrics
    total_trades = 0
    winning_trades = 0
    max_drawdown = 0
    peak_value = portfolio_value
    
    # Iterate through each day
    for i in range(1, len(data)):
        current_price = data['close'].iloc[i]
        prev_price = data['close'].iloc[i-1]
        
        # Get alpha signal for current day
        alpha_signal = combined_alpha.iloc[i][ticker]
        
        # Calculate technical signals
        momentum_signal = data['sma_20'].iloc[i] > data['sma_50'].iloc[i]
        value_signal = current_price < data['bb_lower'].iloc[i]
        rsi_signal = data['rsi'].iloc[i] < 30
        macd_signal = data['macd'].iloc[i] > data['signal'].iloc[i]
        
        # Count positive signals
        signals = [momentum_signal, value_signal, rsi_signal, macd_signal]
        signal_strength = sum(signals)
        
        # Calculate position size based on volatility and signal strength
        volatility = data['volatility'].iloc[i]
        position_size = calculate_position_size(volatility, signal_strength)
        
        # Trading logic
        if position == 0:  # Not in a position
            if signal_strength >= 2:  # At least 2 positive signals
                position = 1
                entry_price = current_price
                entry_date = data.index[i]
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'position_size': position_size
                })
        
        elif position == 1:  # In a position
            # Calculate position return
            position_return = (current_price - entry_price) / entry_price
            cumulative_return += position_return
            
            # Check stop loss and take profit
            if position_return <= stop_loss or position_return >= take_profit:
                position = 0
                trades[-1].update({
                    'exit_date': data.index[i],
                    'exit_price': current_price,
                    'return': position_return
                })
                if position_return > 0:
                    winning_trades += 1
                total_trades += 1
                cumulative_return = 0
            # Check signal strength for exit
            elif signal_strength < 2:
                position = 0
                trades[-1].update({
                    'exit_date': data.index[i],
                    'exit_price': current_price,
                    'return': position_return
                })
                if position_return > 0:
                    winning_trades += 1
                total_trades += 1
                cumulative_return = 0
        
        # Update portfolio value
        if position == 1:
            portfolio_value *= (1 + position_return * position_size)
        
        portfolio_values.append(portfolio_value)
        
        # Update maximum drawdown
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        current_drawdown = (peak_value - portfolio_value) / peak_value
        max_drawdown = max(max_drawdown, current_drawdown)
    
    # Calculate final metrics
    total_return = (portfolio_value - 100000) / 100000
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    sharpe_ratio = np.sqrt(252) * np.mean(data['returns']) / np.std(data['returns']) if len(data['returns']) > 0 else 0
    
    # Add positions and portfolio value to data
    data['position'] = [1 if t.get('entry_date') == d else 0 for d in data.index for t in trades]
    data['portfolio_value'] = portfolio_values
    
    # Calculate metrics
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'average_position_size': np.mean([t['position_size'] for t in trades]) if trades else 0,
        'signal_counts': {
            'momentum': sum(data['sma_20'] > data['sma_50']),
            'value': sum(data['close'] < data['bb_lower']),
            'rsi': sum(data['rsi'] < 30),
            'macd': sum(data['macd'] > data['signal'])
        },
        'average_signal_strength': np.mean([sum([data['sma_20'].iloc[i] > data['sma_50'].iloc[i],
                                                data['close'].iloc[i] < data['bb_lower'].iloc[i],
                                                data['rsi'].iloc[i] < 30,
                                                data['macd'].iloc[i] > data['signal'].iloc[i]])
                                          for i in range(len(data))])
    }
    
    return data, metrics

def plot_results(data: pd.DataFrame, ticker: str, metrics: Dict):
    """
    Plot the backtest results
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with positions and returns
    ticker : str
        Ticker symbol
    metrics : Dict
        Dictionary with performance metrics
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot price and indicators
    ax1.plot(data.index, data['close'], label='Price', color='blue')
    ax1.plot(data.index, data['sma_20'], label='SMA 20', color='orange')
    ax1.plot(data.index, data['sma_50'], label='SMA 50', color='green')
    ax1.plot(data.index, data['sma_200'], label='SMA 200', color='red')
    ax1.set_title(f'{ticker} Price and Moving Averages')
    ax1.legend()
    
    # Plot technical indicators
    ax2.plot(data.index, data['rsi'], label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_title('Technical Indicators')
    ax2.legend()
    
    # Plot portfolio value and position size
    ax3.plot(data.index, data['portfolio_value'], label='Portfolio Value', color='blue')
    ax3.set_title('Portfolio Performance')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print(f"\nPerformance Metrics for {ticker}:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Average Position Size: {metrics['average_position_size']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print("\nSignal Analysis:")
    print(f"Momentum Signals: {metrics['signal_counts']['momentum']}")
    print(f"Value Signals: {metrics['signal_counts']['value']}")
    print(f"RSI Signals: {metrics['signal_counts']['rsi']}")
    print(f"MACD Signals: {metrics['signal_counts']['macd']}")
    print(f"Average Signal Strength: {metrics['average_signal_strength']:.2f}")

def run_backtest():
    """
    Run backtest for all tickers
    """
    # Load data
    data_dir = "data"
    data_files = [f for f in os.listdir(data_dir) if f.startswith('backtest_data_')]
    
    # Initialize alpha factors
    alpha_factors = AlphaFactors({})
    
    for file in data_files:
        ticker = file.split('_')[2]
        print(f"\nProcessing {ticker}...")
        
        # Load data
        data = pd.read_csv(os.path.join(data_dir, file), index_col=0, parse_dates=True)
        
        # Update alpha factors data
        alpha_factors.data[ticker] = data
        
        # Run backtest
        results, metrics = backtest_strategy(ticker, data, alpha_factors)
        
        # Plot results
        plot_results(results, ticker, metrics)

if __name__ == "__main__":
    run_backtest() 