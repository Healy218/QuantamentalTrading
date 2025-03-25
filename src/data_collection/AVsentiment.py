import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from dotenv import load_dotenv
from AVtrending import get_trending_stocks

def get_stock_sentiment():
    """
    Fetches trending stocks and analyzes their sentiment, saving results to a database.
    """
    # Load environment variables
    load_dotenv(dotenv_path="C:/Users/mrhea/OneDrive/Documents/Coding Projects/Stonks/.env")
    
    # Get API key from environment variables
    api_key = os.getenv('ALPHAVANTAGE_KEY')
    
    # Get trending stocks
    gainers, losers, active = get_trending_stocks()
    
    # Combine all tickers into a single list
    all_tickers = list(set(
        gainers['ticker'].tolist() + 
        losers['ticker'].tolist() + 
        active['ticker'].tolist()
    ))
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Get current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Initialize list to store all sentiment data
    all_sentiment_data = []
    
    # Process each ticker
    for ticker in all_tickers:
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        
        if 'feed' in data:
            for article in data['feed']:
                sentiment_data = {
                    'ticker': ticker,
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'time_published': article.get('time_published', ''),
                    'summary': article.get('summary', ''),
                    'overall_sentiment_score': article.get('overall_sentiment_score', 0),
                    'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                    'ticker_sentiment': article.get('ticker_sentiment', []),
                    'timestamp': current_time
                }
                all_sentiment_data.append(sentiment_data)
    
    # Convert to DataFrame
    sentiment_df = pd.DataFrame(all_sentiment_data)
    
    # Save to CSV
    sentiment_file = os.path.join(data_dir, f'stock_sentiment_{current_time}.csv')
    sentiment_df.to_csv(sentiment_file, index=False)
    
    print(f"Sentiment data saved to: {sentiment_file}")
    return sentiment_df

if __name__ == "__main__":
    # Example usage
    sentiment_data = get_stock_sentiment()
    print("\nSample of sentiment data:")
    print(sentiment_data.head())