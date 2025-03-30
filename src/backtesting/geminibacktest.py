import pandas as pd
import numpy as np
import yfinance as yf
import talib  # For technical indicators
import backtrader as bt  # For backtesting
from datetime import datetime
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk



# Custom PandasData class to handle our indicators
class CustomPandasData(bt.feeds.PandasData):
    lines = ('Signal',)  # Add Signal as a line
    params = (
        ('datetime', None),  # Use index as datetime
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('Signal', 'Signal'),  # Map Signal column to Signal line
    )

# 1. Data Acquisition [cite: 141]

def get_spy_data(start_date, end_date, interval):
    """
    Fetches historical SPY data with options data.
    """
    spy_data = yf.download("SPY", start=start_date, end=end_date, interval=interval)
    
    # Check if we got any data
    if spy_data.empty:
        raise ValueError("No data was downloaded from Yahoo Finance")
    
    # Flatten the MultiIndex columns
    spy_data.columns = [col[0] for col in spy_data.columns]
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in spy_data.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Available columns: {spy_data.columns}")
    
    return spy_data

# 2. Feature Engineering [cite: 142]

def calculate_indicators(df):
    """
    Calculates technical indicators.
    """
    # Ensure we have data
    if df.empty:
        raise ValueError("Empty DataFrame provided")
    
    # Convert Close price to numpy array and ensure it's 1-dimensional
    close_array = df['Close'].values.flatten()
    
    # Calculate indicators
    df['EMA_20'] = talib.EMA(close_array, timeperiod=20)
    df['RSI'] = talib.RSI(close_array, timeperiod=14)
    macd, signal, _ = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal'] = signal
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close_array, timeperiod=20)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    
    return df

# 3. Signal Generation Logic [cite: 145]

def generate_signals(df):
    """
    Generates trading signals based on indicators.
    """
    df['Signal'] = 0  # 0 = No signal
    # Buy signal: RSI crosses below 30 and MACD crosses above signal line
    df.loc[(df['RSI'] < 30) & (df['MACD'] > df['Signal']), 'Signal'] = 1
    # Sell signal: RSI crosses above 70 or MACD crosses below signal line
    df.loc[(df['RSI'] > 70) | (df['MACD'] < df['Signal']), 'Signal'] = -1
    return df

# 4. Backtesting Framework [cite: 146]

class OptionStrategy(bt.Strategy):
    """
    Backtesting strategy for 0DTE options.
    """
    def __init__(self):
        self.signal = self.datas[0].Signal  # Now this will work with our custom data feed
        self.buy_sell_dates = []  # To track buy/sell dates
        self.buy_sell_prices = []  # To track buy/sell prices
        self.portfolio_values = []  # To track portfolio values over time

    def next(self):
        # Record the current portfolio value
        self.portfolio_values.append(self.broker.getvalue())

        if self.signal[0] == 1:  # Buy signal
            self.buy()  # This will need significant refinement for real options trading
            self.buy_sell_dates.append(self.datas[0].datetime.date(0))  # Record date
            self.buy_sell_prices.append(self.datas[0].close[0])  # Record price
        elif self.signal[0] == -1:  # Sell signal
            self.sell()  # This will need significant refinement for real options trading
            self.buy_sell_dates.append(self.datas[0].datetime.date(0))  # Record date
            self.buy_sell_prices.append(self.datas[0].close[0])  # Record price

# Plotting all the data and indicators             
def plot_signals_and_indicators(df, portfolio_values, buy_sell_dates, buy_sell_prices):
    """
    Plots the closing price, indicators, signals, and portfolio value.
    """
    plt.figure(figsize=(14, 16))

    # Plot Closing Price
    plt.subplot(4, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')  # Use df.index for dates
    plt.title('Close Price')
    plt.legend()

    # Plot Indicators
    plt.subplot(4, 1, 2)
    plt.plot(df.index, df['EMA_20'], label='EMA 20', color='orange')  # Use df.index for dates
    plt.plot(df.index, df['BB_Upper'], label='Bollinger Upper', color='red', linestyle='--')  # Use df.index for dates
    plt.plot(df.index, df['BB_Lower'], label='Bollinger Lower', color='green', linestyle='--')  # Use df.index for dates
    plt.title('Technical Indicators')
    plt.legend()

    # Plot Signals
    plt.subplot(4, 1, 3)
    plt.plot(df.index, df['Signal'], label='Signal', color='purple', marker='o', markersize=5)  # Use df.index for dates
    plt.title('Trading Signals')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.legend()

    # Plot Portfolio Value
    plt.subplot(4, 1, 4)
    plt.plot(df.index, portfolio_values, label='Portfolio Value', color='green')  # Use df.index for dates
    plt.title('Portfolio Value Over Time')
    
    # Mark Buy and Sell Points
    for date, price in zip(buy_sell_dates, buy_sell_prices):
        plt.scatter(date, price, color='red' if price < portfolio_values[-1] else 'blue', marker='o', s=100, label='Buy/Sell' if price < portfolio_values[-1] else 'Sell')

    plt.legend()
    plt.tight_layout(pad=3.0)
    plt.show()
            

if __name__ == '__main__':
    start_date = "2025-02-01"  # Changed to past date since future data won't exist
    end_date = "2025-03-28"    # Changed to past date since future data won't exist
    interval = '2m'            # 1m, 5m, 15m, 30m, 60m, 90m, 1h, 1d
    start_capital = 10000.0    # Initial capital

    try:
        # Data Acquisition
        print("Downloading SPY data...")
        spy_data = get_spy_data(start_date, end_date, interval)
        print(f"Downloaded {len(spy_data)} data points")

        # Feature Engineering
        print("Calculating indicators...")
        spy_data = calculate_indicators(spy_data)

        # Signal Generation
        print("Generating signals...")
        spy_data = generate_signals(spy_data)

        # Backtesting
        cerebro = bt.Cerebro()
        data = CustomPandasData(dataname=spy_data)  # Use our custom data feed
        cerebro.adddata(data)
        cerebro.addstrategy(OptionStrategy)

        # Initial capital (Important: Adjust this based on your strategy)
        cerebro.broker.setcash(10000.0)
        cerebro.addsizer(bt.sizers.FixedSize, stake=1)  # Very basic sizing

        # Run the backtest
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
        strategies = cerebro.run()
        strategy = strategies[0]

        # Get the strategy instance to access portfolio values and buy/sell data
        # strategy = cerebro.strategies[0]
        portfolio_values = strategy.portfolio_values
        buy_sell_dates = strategy.buy_sell_dates
        buy_sell_prices = strategy.buy_sell_prices

        print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')

        # Plotting the indicators, signals, and portfolio value
        plot_signals_and_indicators(spy_data, portfolio_values, buy_sell_dates, buy_sell_prices)  # Call the updated plotting function

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("DataFrame info:")
        print(spy_data.info())