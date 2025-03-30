import pandas as pd
import numpy as np
import yfinance as yf
import talib
import backtrader as bt
import pytz  # Import the pytz library for timezone handling
import mplfinance as mpf
import matplotlib.pyplot as plt
import tkinter as tk
from datetime import datetime, time
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from strategies.momenturmburst import MomentumBurstStrategy
from backtrader import indicators as btind
from strategies.orb import OpeningRangeBreakoutStrategy
from strategies.reversal import ReversalAtKeyLevelsStrategy
from strategies.rsidivergence import RSIDivergenceStrategy
from strategies.volumebreakout import VolumeBreakoutStrategy
from strategies.macdzerocross import MACDZeroCrossStrategy

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

# 1. Data Acquisition

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

    # Make the index timezone-aware (UTC) if it isn't already
    if spy_data.index.tz is None:
        spy_data.index = spy_data.index.tz_localize('UTC')
    else:
        spy_data.index = spy_data.index.tz_convert('UTC')

    return spy_data

"""
def filter_trading_hours(df):
    
    Filters a DataFrame to include only standard US trading hours (9:30 AM to 4:00 PM ET)
    and excludes weekends. Assumes the DataFrame index needs to be in UTC.
    
    # Ensure index is timezone-aware and convert to UTC
    if df.index.tz is None:
        df = df.copy() # Create a copy to avoid modifying the original unexpectedly
        df.index = df.index.tz_localize('UTC')
    else:
        df = df.copy()
        df.index = df.index.tz_convert('UTC')

    # Define trading hours in UTC (9:30 AM to 4:00 PM EDT)
    start_time_utc = time(13, 30)  # 9:30 AM EDT (UTC-4 during standard time)
    end_time_utc = time(20, 0)    # 4:00 PM EDT (UTC-4 during standard time)

    # Filter by time
    filtered_df = df.between_time(start_time_utc, end_time_utc)

    # Filter out weekends (Saturday: 5, Sunday: 6 in pandas datetime)
    filtered_df = filtered_df[filtered_df.index.dayofweek < 5]

    return filtered_df
"""

# 2. Feature Engineering

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

# 3. Signal Generation Logic

def generate_signals(df):
    """
    Generates trading signals based on indicators using crossovers.
    """
    df['Signal'] = 0  # 0 = No signal

    # Buy signal: RSI crosses above 35 or MACD crosses above signal line
    buy_condition_rsi = (df['RSI'].shift(1) <= 45) & (df['RSI'] > 45)
    buy_condition_macd = (df['MACD'].shift(1) <= df['Signal'].shift(1)) & (df['MACD'] > df['Signal'])
    df.loc[buy_condition_rsi | buy_condition_macd, 'Signal'] = 1

    # Sell signal: RSI crosses below 65 or MACD crosses below signal line
    sell_condition_rsi = (df['RSI'].shift(1) >= 65) & (df['RSI'] < 65)
    sell_condition_macd = (df['MACD'].shift(1) >= df['Signal'].shift(1)) & (df['MACD'] < df['Signal'])
    df.loc[sell_condition_rsi | sell_condition_macd, 'Signal'] = -1

    return df

# 4. Backtesting Framework

class OptionStrategy(bt.Strategy):
    """
    Backtesting strategy for 0DTE options.
    """
    def __init__(self):
        self.signal = self.datas[0].Signal
        self.buy_sell_dates = []
        self.buy_sell_prices = []
        self.portfolio_values = []
        self.timezone = pytz.utc
        self.trade_state = 0  # 0: no position, 1: long, -1: short

    def next(self):
        self.portfolio_values.append(self.broker.getvalue())
        current_datetime = self.datas[0].datetime.datetime(0)
        timezone_aware_datetime = pd.Timestamp(current_datetime, tz='UTC')
        current_price = self.datas[0].close[0]

        if not self.trade_state:  # If not in a position
            if self.signal[0] == 1:  # Buy signal
                self.buy()
                self.trade_state = 1
                self.buy_sell_dates.append(timezone_aware_datetime)
                self.buy_sell_prices.append(current_price)
                print(f"Buy order opened on: {timezone_aware_datetime} at {current_price}")
            elif self.signal[0] == -1:  # Sell signal
                self.sell()
                self.trade_state = -1
                self.buy_sell_dates.append(timezone_aware_datetime)
                self.buy_sell_prices.append(current_price)
                print(f"Sell order opened on: {timezone_aware_datetime} at {current_price}")

        elif self.trade_state == 1:  # If in a long position
            if self.signal[0] == -1:  # Sell signal to close long position
                self.close()  # Backtrader's close method will close the current position
                self.trade_state = 0
                self.buy_sell_dates.append(timezone_aware_datetime)
                self.buy_sell_prices.append(current_price)
                print(f"Long position closed on: {timezone_aware_datetime} at {current_price}")

        elif self.trade_state == -1:  # If in a short position
            if self.signal[0] == 1:  # Buy signal to close short position
                self.close()  # Backtrader's close method will close the current position
                self.trade_state = 0
                self.buy_sell_dates.append(timezone_aware_datetime)
                self.buy_sell_prices.append(current_price)
                print(f"Short position closed on: {timezone_aware_datetime} at {current_price}")


# 5. Plotting
def plot_signals_and_indicators(df_plot, portfolio_values, buy_sell_dates, buy_sell_prices, 
                              start_date=None, end_date=None, figscale=1.5):
    """
    Plots the closing price, indicators, signals, and portfolio value using mplfinance with flexible sizing.
    
    Args:
        df_plot (pd.DataFrame): DataFrame containing price data and indicators
        portfolio_values (list): List of portfolio values from backtesting
        buy_sell_dates (list): List of datetime objects for buy/sell signals
        buy_sell_prices (list): List of prices corresponding to buy/sell signals
        start_date (str, optional): Start date for plotting (e.g., '2025-02-01')
        end_date (str, optional): End date for plotting (e.g., '2025-03-28')
        figscale (float): Base figure scaling factor
    """
    # Slice data based on date range if provided
    if start_date or end_date:
        df_plot = df_plot.loc[start_date:end_date].copy()
        portfolio_values = portfolio_values[:len(df_plot)]

    # Calculate dynamic figure size based on data length
    num_points = len(df_plot)
    base_width = min(15, max(10, num_points / 50))  # Width between 10-15 inches
    base_height = 8  # Base height for main chart + panels
    dynamic_figsize = (base_width * figscale, base_height * figscale)

    # Prepare buy and sell signals
    buy_mask = df_plot['Signal'] == 1
    sell_mask = df_plot['Signal'] == -1
    buy_prices = pd.Series(np.nan, index=df_plot.index)
    sell_prices = pd.Series(np.nan, index=df_plot.index)
    buy_prices[buy_mask] = df_plot['Close'][buy_mask]
    sell_prices[sell_mask] = df_plot['Close'][sell_mask]

    # Calculate MACD limits before creating plots
    macd_min = min(df_plot['MACD'].min(), df_plot['Signal'].min())
    macd_max = max(df_plot['MACD'].max(), df_plot['Signal'].max())
    macd_padding = (macd_max - macd_min) * 0.1
    macd_ylim = (macd_min - macd_padding, macd_max + macd_padding)

    # Prepare additional plots with labels
    apds = [
        # RSI
        mpf.make_addplot(df_plot['RSI'], panel=1, color='purple', ylabel='RSI', 
                        ylim=(0, 100)),
        # MACD
        mpf.make_addplot(df_plot['MACD'], panel=2, color='blue', ylabel='MACD', 
                        ylim=macd_ylim),
        mpf.make_addplot(df_plot['Signal'], panel=2, color='orange', 
                        ylim=macd_ylim),
        # Bollinger Bands and EMA
        mpf.make_addplot(df_plot['BB_Upper'], panel=0, color='gray', linestyle='--'),
        mpf.make_addplot(df_plot['BB_Lower'], panel=0, color='gray', linestyle='--'),
        mpf.make_addplot(df_plot['EMA_20'], panel=0, color='red', linestyle='-'),
        # Buy/Sell Signals
        mpf.make_addplot(buy_prices, type='scatter', markersize=100, marker='^', 
                        color='green', panel=0),
        mpf.make_addplot(sell_prices, type='scatter', markersize=100, marker='v', 
                        color='red', panel=0),
    ]

    # Portfolio value plot
    portfolio_df = pd.Series(portfolio_values[:len(df_plot)], index=df_plot.index)
    portfolio_min, portfolio_max = portfolio_df.min(), portfolio_df.max()
    portfolio_padding = (portfolio_max - portfolio_min) * 0.1
    apds.append(mpf.make_addplot(portfolio_df, panel=3, color='green', 
                                ylabel='Portfolio Value',
                                ylim=(portfolio_min - portfolio_padding, 
                                      portfolio_max + portfolio_padding)))

    # Create the main plot
    fig, axes = mpf.plot(
        df_plot,
        type='candle',
        style='yahoo',
        title=f'SPY Trading Strategy ({df_plot.index[0].date()} to {df_plot.index[-1].date()})',
        ylabel='Price ($)',
        volume=True,
        volume_panel=4,  # Explicitly set volume panel
        addplot=apds,
        figscale=figscale,
        figsize=dynamic_figsize,
        panel_ratios=(3, 1, 1, 1, 1),  # 5 panels including volume
        tight_layout=True,
        returnfig=True
    )

    # Customize axes labels and add legend
    axes[0].set_ylabel('Price ($)')  # Main chart
    axes[1].set_ylabel('RSI')        # RSI panel
    axes[2].set_ylabel('MACD')       # MACD panel
    axes[3].set_ylabel('Portfolio')  # Portfolio panel
    axes[4].set_ylabel('Volume')     # Volume panel

    # Add RSI reference lines
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)

    # Add legend to main chart for signals and indicators
    axes[0].legend(['Price', 'BB Upper', 'BB Lower', 'EMA 20', 'Buy', 'Sell'], 
                  loc='upper left')

    # Add legend to MACD panel
    axes[2].legend(['MACD', 'Signal'], loc='upper left')

    # Improve date formatting based on data range
    date_range = (df_plot.index[-1] - df_plot.index[0]).days
    if date_range < 7:
        mpf.plot(df_plot, type='candle', ax=axes[0], datetime_format='%H:%M')
    elif date_range < 30:
        mpf.plot(df_plot, type='candle', ax=axes[0], datetime_format='%m-%d')

    # Display the plot
    plt.show()

# Update the show_plots function to allow date range parameters
def show_plots(df, portfolio_values, buy_sell_dates, buy_sell_prices, 
              start_date=None, end_date=None, figscale=1.5):
    """
    Helper function to call the plotting function with optional date range
    """
    plot_signals_and_indicators(df, portfolio_values, buy_sell_dates, buy_sell_prices, 
                              start_date, end_date, figscale)

# Note: You'll need to fix the commented-out filter_trading_hours function call in your main block
# by uncommenting it and using it correctly. Here's the corrected main block portion:

if __name__ == '__main__':
    start_date = "2025-02-01"
    end_date = "2025-03-28"
    interval = '5m'
    start_capital = 10000.0

    try:
        # Data Acquisition
        print("Downloading SPY data...")
        spy_data = get_spy_data(start_date, end_date, interval)
        print(f"Downloaded {len(spy_data)} data points")

        # Feature Engineering (for original OptionStrategy)
        print("Calculating indicators...")
        spy_data = calculate_indicators(spy_data.copy())

        # Signal Generation (for original OptionStrategy)
        print("Generating signals...")
        spy_data = generate_signals(spy_data.copy())

        # Backtesting
        cerebro = bt.Cerebro()
        data = CustomPandasData(dataname=spy_data)
        cerebro.adddata(data)

         # Choose strategy
        strategy_choice = "original"  # Options: "original", "momentum", "orb", "macd", "reversal", "rsi", "volume"
        
        if strategy_choice == "original":
            cerebro.addstrategy(OptionStrategy)
        elif strategy_choice == "momentum":
            cerebro.addstrategy(MomentumBurstStrategy)
        elif strategy_choice == "orb":
            cerebro.addstrategy(OpeningRangeBreakoutStrategy)
        elif strategy_choice == "macd":
            cerebro.addstrategy(MACDZeroCrossStrategy)
        elif strategy_choice == "reversal":
            cerebro.addstrategy(ReversalAtKeyLevelsStrategy)
        elif strategy_choice == "rsi":
            cerebro.addstrategy(RSIDivergenceStrategy)
        elif strategy_choice == "volume":
            cerebro.addstrategy(VolumeBreakoutStrategy)
        else:
            raise ValueError("Invalid strategy choice")

        cerebro.broker.setcash(start_capital)
        cerebro.addsizer(bt.sizers.FixedSize, stake=100)
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
        strategies = cerebro.run()
        strategy = strategies[0]

        # Extract results (modify based on strategy)
        portfolio_values = strategy.portfolio_values if strategy_choice == "original" else [cerebro.broker.getvalue()] * len(spy_data)
        buy_sell_dates = strategy.buy_sell_dates if strategy_choice == "original" else []
        buy_sell_prices = strategy.buy_sell_prices if strategy_choice == "original" else []

        print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')

        # Plotting
        show_plots(spy_data, portfolio_values, buy_sell_dates, buy_sell_prices,
                  start_date="2025-02-01", 
                  end_date="2025-03-28",
                  figscale=1.5)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()