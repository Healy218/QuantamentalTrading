import yfinance as yf
import talib
import backtrader as bt
import pandas as pd

# Define the trading strategy
class EndOfDayAlpha(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14) # Initialize RSI indicator
        self.macd = bt.indicators.MACD(self.data.close) # Initialize MACD indicator

    def next(self):
        # Check if it's the last 30 minutes of the trading day
        current_time = self.data.datetime.time(0)
        end_of_day = pd.to_datetime('16:00:00').time()
        start_of_last_30_min = pd.to_datetime('15:30:00').time()

        if start_of_last_30_min <= current_time <= end_of_day:
            # Generate buy signals based on conditions
            if self.rsi[0] < 30 and self.macd.macd[0] > self.macd.signal[0]:
                self.buy()  # Buy if RSI is oversold and MACD is bullish
            # Add more conditions here as needed

# Data Acquisition
# Ticker and date range
ticker = "SPY"
start_date = "2025-03-23"  # Adjust as needed
end_date = "2025-03-28"    # Adjust as needed

# Fetch minute-level data using yfinance
data = yf.download(ticker, start=start_date, end=end_date, interval="1m")

# Backtrader Setup
cerebro = bt.Cerebro()

# Create a Backtrader data feed from the yfinance data
feed = bt.feeds.PandasData(dataname=data)
cerebro.adddata(feed)

cerebro.addstrategy(EndOfDayAlpha) # Add the strategy

cerebro.broker.setcash(100000.0)  # Set initial capital
cerebro.addsizer(bt.sizers.FixedSize, stake=100) # Set position size

print('Starting Portfolio Value:', cerebro.broker.getvalue())
cerebro.run() # Run the backtest
print('Final Portfolio Value:', cerebro.broker.getvalue())

# cerebro.plot()  # Plot the backtest results (optional, may require GUI setup)