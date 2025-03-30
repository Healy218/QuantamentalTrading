from AlphaVantage import fetch_stock_data, process_stock_data
from AVtrending import get_trending_stocks
from AVsentiment import get_stock_sentiment
import pandas as pd
import time
from datetime import datetime
import os

def collect_backtest_data():
    """
    Collect historical data and sentiment for trending tickers to prepare for backtesting
    Returns a dictionary of DataFrames with historical data for each ticker
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Get trending tickers from Alpha Vantage
    gainers, losers, active = get_trending_stocks()
    
    # Combine all tickers into a single list
    stock_symbols = list(set(
        gainers['ticker'].tolist() + 
        losers['ticker'].tolist() + 
        active['ticker'].tolist()
    ))
    
    print(f"Collecting data for {len(stock_symbols)} trending tickers...")
    
    # Dictionary to store historical data for each ticker
    historical_data = {}
    
    # Get sentiment data for all tickers
    sentiment_data = get_stock_sentiment()
    
    for symbol in stock_symbols:
        try:
            print(f"Fetching data for {symbol}...")
            # Fetch the raw JSON data from Alpha Vantage
            data = fetch_stock_data(symbol=symbol)
            
            # Check if we got valid data
            if not data or 'Error Message' in data:
                print(f"No data available for {symbol}")
                continue
                
            # Check if we have time series data
            time_series_keys = [key for key in data.keys() if 'Time Series' in key]
            if not time_series_keys:
                print(f"No time series data available for {symbol}")
                continue
            
            # Process the JSON into a pandas DataFrame
            df = process_stock_data(data)
            
            # Skip if we got an empty DataFrame
            if df.empty:
                print(f"Empty data received for {symbol}")
                continue
            
            # Add sentiment data for this ticker if available
            ticker_sentiment = sentiment_data[sentiment_data['ticker'] == symbol]
            if not ticker_sentiment.empty:
                df['sentiment_score'] = ticker_sentiment['overall_sentiment_score'].iloc[0]
                df['sentiment_label'] = ticker_sentiment['overall_sentiment_label'].iloc[0]
            
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
        if 'sentiment_score' in df.columns:
            print(f"Sentiment: {df['sentiment_label'].iloc[0]} (Score: {df['sentiment_score'].iloc[0]})")