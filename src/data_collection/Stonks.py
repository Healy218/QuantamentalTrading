from AlphaVantage import fetch_stock_data, process_stock_data
from StockTwitsFileMaker import get_trending_tickers
import pandas as pd
import time
from datetime import datetime

def collect_backtest_data():
    """
    Collect historical data for trending tickers to prepare for backtesting
    Returns a dictionary of DataFrames with historical data for each ticker
    """
    # Get trending tickers from StockTwits
    stock_symbols = get_trending_tickers()
    print(f"Collecting data for {len(stock_symbols)} trending tickers...")
    
    # Dictionary to store historical data for each ticker
    historical_data = {}
    
    for symbol in stock_symbols:
        try:
            print(f"Fetching data for {symbol}...")
            # Fetch the raw JSON data from Alpha Vantage
            data = fetch_stock_data(symbol=symbol)
            # Process the JSON into a pandas DataFrame
            df = process_stock_data(data)
            
            # Store the DataFrame in our dictionary
            historical_data[symbol] = df
            
            # Add a delay to respect API rate limits
            time.sleep(12)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    # Save the data to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for symbol, df in historical_data.items():
        filename = f"data/backtest_data_{symbol}_{timestamp}.csv"
        df.to_csv(filename)
        print(f"Saved data for {symbol} to {filename}")
    
    return historical_data

if __name__ == '__main__':
    # Collect data for backtesting
    historical_data = collect_backtest_data()
    
    # Print summary of collected data
    print("\nData Collection Summary:")
    for symbol, df in historical_data.items():
        print(f"{symbol}: {len(df)} data points from {df.index.min()} to {df.index.max()}")