import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from dotenv import load_dotenv

def get_trending_stocks():
    """
    Fetches and returns the top 5 gainers, 5 losers, and 10 most active stocks from Alpha Vantage.
    
    Returns:
        tuple: (top_gainers_df, top_losers_df, top_active_df)
    """
    # Load environment variables
    load_dotenv(dotenv_path="C:/Users/mrhea/OneDrive/Documents/Coding Projects/Stonks/.env")
        
    # Get API key from environment variables
    api_key = os.getenv('ALPHAVANTAGE_KEY')
    url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    # Get current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Process gainers, losers, and most active into DataFrames
    gainers_df = pd.DataFrame(data['top_gainers'])
    losers_df = pd.DataFrame(data['top_losers'])
    most_active_df = pd.DataFrame(data['most_actively_traded'])

    # Add timestamp column to all DataFrames
    gainers_df['timestamp'] = current_time
    losers_df['timestamp'] = current_time
    most_active_df['timestamp'] = current_time

    # Get top 5 gainers and losers, top 10 most active
    top_gainers = gainers_df.head(5)
    top_losers = losers_df.head(5)
    top_active = most_active_df.head(10)

    return top_gainers, top_losers, top_active

if __name__ == "__main__":
    # Example usage
    gainers, losers, active = get_trending_stocks()
    print("\nTop 5 Gainers:")
    print(gainers)
    print("\nTop 5 Losers:")
    print(losers)
    print("\nTop 10 Most Active:")
    print(active)