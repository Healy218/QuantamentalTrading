import pandas as pd
import numpy as np
import yfinance as yf
import talib
import backtrader as bt
from datetime import datetime
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pytz  # Import the pytz library for timezone handling


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

    # Buy signal: RSI crosses above 35 and MACD crosses above signal line
    buy_condition_rsi = (df['RSI'].shift(1) <= 35) & (df['RSI'] > 35)
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
                
                
# Plotting all the data and indicators
def plot_signals_and_indicators(df, portfolio_values, buy_sell_dates, buy_sell_prices):
    """
    Plots the closing price, indicators, signals, and portfolio value.
    """
    # Create a new figure for each plot
    fig = plt.Figure(figsize=(12, 10), tight_layout=True)

    # Plot Closing Price
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.set_title('Close Price')
    ax1.legend()

    # Plot EMA and Bollinger Bands
    ax2 = fig.add_subplot(5, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['EMA_20'], label='EMA 20', color='orange')
    ax2.plot(df.index, df['BB_Upper'], label='Bollinger Upper', color='red', linestyle='--')
    ax2.plot(df.index, df['BB_Lower'], label='Bollinger Lower', color='green', linestyle='--')
    ax2.set_title('EMA 20 and Bollinger Bands')
    ax2.legend()

    # Plot RSI
    ax3 = fig.add_subplot(5, 1, 3, sharex=ax1)
    ax3.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax3.axhline(30, color='gray', linestyle='--', linewidth=0.5)
    ax3.axhline(70, color='gray', linestyle='--', linewidth=0.5)
    ax3.axhline(50, color='black', linestyle='--', linewidth=0.5) # Added line at 50 for reference
    ax3.set_title('RSI')
    ax3.legend()

    # Plot MACD
    ax4 = fig.add_subplot(5, 1, 4, sharex=ax1)
    ax4.plot(df.index, df['MACD'], label='MACD', color='red')
    ax4.plot(df.index, df['Signal'], label='Signal Line', color='blue')
    ax4.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax4.set_title('MACD')
    ax4.legend()

    # Plot Portfolio Value
    ax5 = fig.add_subplot(5, 1, 5, sharex=ax1)
    ax5.plot(df.index, portfolio_values, label='Portfolio Value', color='green')
    ax5.set_title('Portfolio Value Over Time')

    # Mark Buy and Sell Points on the Price Chart
    for date, price in zip(buy_sell_dates, buy_sell_prices):
        signal_value = df.loc[date, 'Signal']
        color = 'green' if signal_value == 1 else 'red'
        ax1.scatter(date, price, color=color, marker='^' if signal_value == 1 else 'v', s=100) # Changed marker

    return fig

def show_plots(df, portfolio_values, buy_sell_dates, buy_sell_prices):
    root = tk.Tk()
    root.title("Trading Strategy Plots")

    # Create a frame for the plots and the table
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas for the plots
    plot_canvas = tk.Canvas(main_frame)
    plot_frame = ttk.Frame(plot_canvas)
    plot_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=plot_canvas.yview)
    plot_canvas.configure(yscrollcommand=plot_scrollbar.set)

    plot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    plot_canvas.create_window((0, 0), window=plot_frame, anchor="nw")

    # Add the plots to the plot frame
    fig = plot_signals_and_indicators(df, portfolio_values, buy_sell_dates, buy_sell_prices)
    canvas_agg = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_agg.draw()
    canvas_agg.get_tk_widget().pack()

    # Create a table for buy/sell data
    table_frame = ttk.Frame(main_frame)
    table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Create Treeview for the table
    tree = ttk.Treeview(table_frame, columns=("Date", "Price", "Action"), show='headings')
    tree.heading("Date", text="Date")
    tree.heading("Price", text="Price")
    tree.heading("Action", text="Action")

    print("\nInside show_plots:")
    print("Number of buy_sell_dates:", len(buy_sell_dates))
    for i, (date, price) in enumerate(zip(buy_sell_dates, buy_sell_prices)):
        try:
            signal = df.loc[date, 'Signal']
            action = "Unknown"
            if signal == 1:
                action = "Buy"
            elif signal == -1:
                action = "Sell"
            print(f"Processing index {i}: Date={date}, Price={price}, Signal={signal}, Action={action}")
            tree.insert("", "end", values=(date, price, action))
        except KeyError as e:
            print(f"KeyError in show_plots at index {i} for date: {date}")
            print(f"Error: {e}")

    tree.pack(fill=tk.BOTH, expand=True)
    # Adjust layout
    plot_frame.bind("<Configure>", lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all")))

    root.mainloop()


if __name__ == '__main__':
    start_date = "2020-02-01"
    end_date = "2025-03-28"
    interval = '1d'
    start_capital = 10000.0

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

        # Check if any signals were generated
        print(f"Number of Buy signals: {spy_data['Signal'].where(spy_data['Signal'] == 1).count()}")
        print(f"Number of Sell signals: {spy_data['Signal'].where(spy_data['Signal'] == -1).count()}")

        # Backtesting
        cerebro = bt.Cerebro()
        data = CustomPandasData(dataname=spy_data)
        cerebro.adddata(data)
        cerebro.addstrategy(OptionStrategy)

        # Initial capital (Important: Adjust this based on your strategy)
        cerebro.broker.setcash(10000.0)
        cerebro.addsizer(bt.sizers.FixedSize, stake=100)  # Very basic sizing

        # Run the backtest
        print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')
        strategies = cerebro.run()
        strategy = strategies[0]
        print(strategy)
        # Get the strategy instance to access portfolio values and buy/sell data
        # strategy = cerebro.strategies[0]
        portfolio_values = strategy.portfolio_values
        buy_sell_dates = strategy.buy_sell_dates
        buy_sell_prices = strategy.buy_sell_prices

        print(f'Final Portfolio Value: {cerebro.broker.getvalue()}')

        # Plotting the indicators, signals, and portfolio value
        show_plots(spy_data, portfolio_values, buy_sell_dates, buy_sell_prices)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("DataFrame info:")
        if 'spy_data' in locals():
            print(spy_data.info())