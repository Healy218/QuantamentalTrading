from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime

# Replace with your Alpaca API keys
API_KEY = "your_api_key"
SECRET_KEY = "your_secret_key"

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Request 1-minute SPY data (e.g., last 30 days)
request_params = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=datetime(2025, 2, 28),  # Adjust start date
    end=datetime(2025, 3, 28)     # Today’s date
)

bars = client.get_stock_bars(request_params).df
df = bars.reset_index().rename(columns={
    "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"
}).set_index("timestamp")

print(df.head())