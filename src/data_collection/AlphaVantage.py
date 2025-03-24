import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from dotenv import load_dotenv

def fetch_stock_data(symbol='IBM', interval='60min'):
    """Fetch stock data from Alpha Vantage API."""
    # Load environment variables
    load_dotenv(dotenv_path="C:/Users/mrhea/OneDrive/Documents/Coding Projects/Stonks/.env")
    
    # Get API key from environment variables
    api_key = os.getenv('ALPHAVANTAGE_KEY')
    
    # Construct the API URL
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={api_key}'
    
    # Make the request
    r = requests.get(url)
    return r.json()

def process_stock_data(data):
    """Convert the JSON data to a pandas DataFrame."""
    # Extract the time series data
    time_series_key = [key for key in data.keys() if 'Time Series' in key][0]
    time_series = data[time_series_key]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Convert string values to float
    for col in df.columns:
        df[col] = df[col].astype(float)
    
    # Rename columns for clarity
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    return df

def plot_stock_data(df, symbol):
    """Create plots for the stock data."""
    # Create a figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price data on the first subplot
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['open'], label='Open Price', color='green', alpha=0.5)
    ax1.fill_between(df.index, df['low'], df['high'], alpha=0.2, color='gray', label='Price Range')
    
    # Format the x-axis with date labels
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Add labels and title for the price subplot
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{symbol} Stock Price')
    ax1.grid(True)
    ax1.legend()
    
    # Plot volume data on the second subplot
    ax2.bar(df.index, df['volume'], color='blue', alpha=0.6)
    
    # Format the x-axis with date labels
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Add labels for the volume subplot
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    
    # Rotate date labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_summary_stats(df):
    """Create a summary of key statistics."""
    # Calculate daily stats
    daily_change = df['close'].iloc[-1] - df['open'].iloc[0]
    daily_change_pct = (daily_change / df['open'].iloc[0]) * 100
    
    # Get daily high and low
    daily_high = df['high'].max()
    daily_low = df['low'].min()
    
    # Create a summary dataframe
    summary = pd.DataFrame({
        'Last Price': [df['close'].iloc[-1]],
        'Open': [df['open'].iloc[0]],
        'Daily Change': [daily_change],
        'Daily Change %': [daily_change_pct],
        'Daily High': [daily_high],
        'Daily Low': [daily_low],
        'Total Volume': [df['volume'].sum()]
    })
    
    return summary

def main():
    # Set the stock symbol
    symbol = input("Enter stock symbol (default: IBM): ") or "IBM"
    
    # Fetch the data
    data = fetch_stock_data(symbol)
    
    # Process the data
    df = process_stock_data(data)
    
    # Create and display summary statistics
    summary = create_summary_stats(df)
    print(f"\n{symbol} Stock Summary:")
    print(summary.to_string(index=False))
    
    # Create the plot
    fig = plot_stock_data(df, symbol)
    
    # Show the plot
    plt.show()
    
    # Option to save the figure
    save = input("\nDo you want to save the chart? (y/n): ")
    if save.lower() == 'y':
        filename = f"{symbol}_stock_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(filename)
        print(f"Chart saved as {filename}")

if __name__ == "__main__":
    main()