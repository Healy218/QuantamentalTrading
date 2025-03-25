import pandas as pd
import os
from datetime import datetime
import glob

def rank_stock_sentiment():
    """
    Processes sentiment data from CSV files and creates a ranking system for tickers.
    Returns a DataFrame with tickers ranked by sentiment from best to worst.
    """
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data')
    
    # Find the most recent sentiment file
    sentiment_files = glob.glob(os.path.join(data_dir, 'stock_sentiment_*.csv'))
    if not sentiment_files:
        raise FileNotFoundError("No sentiment data files found in the data directory")
    
    latest_file = max(sentiment_files, key=os.path.getctime)
    
    # Read the sentiment data
    sentiment_df = pd.read_csv(latest_file)
    
    # Convert sentiment scores to numeric, handling any non-numeric values
    sentiment_df['overall_sentiment_score'] = pd.to_numeric(sentiment_df['overall_sentiment_score'], errors='coerce')
    
    # Calculate average sentiment score per ticker
    ticker_sentiment = sentiment_df.groupby('ticker').agg({
        'overall_sentiment_score': ['mean', 'count'],
        'overall_sentiment_label': lambda x: x.value_counts().index[0]  # Most common sentiment label
    }).reset_index()
    
    # Flatten column names
    ticker_sentiment.columns = ['ticker', 'avg_sentiment_score', 'article_count', 'dominant_sentiment']
    
    # Sort by sentiment score in descending order
    ticker_sentiment = ticker_sentiment.sort_values('avg_sentiment_score', ascending=False)
    
    # Add rank column
    ticker_sentiment['rank'] = range(1, len(ticker_sentiment) + 1)
    
    # Create sentiment category based on score
    def get_sentiment_category(score):
        if score >= 0.5:
            return 'Very Positive'
        elif score >= 0.2:
            return 'Positive'
        elif score >= -0.2:
            return 'Neutral'
        elif score >= -0.5:
            return 'Negative'
        else:
            return 'Very Negative'
    
    ticker_sentiment['sentiment_category'] = ticker_sentiment['avg_sentiment_score'].apply(get_sentiment_category)
    
    # Save the ranking to CSV
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ranking_file = os.path.join(data_dir, f'sentiment_ranking_{current_time}.csv')
    ticker_sentiment.to_csv(ranking_file, index=False)
    
    print(f"Sentiment ranking saved to: {ranking_file}")
    return ticker_sentiment

if __name__ == "__main__":
    # Example usage
    ranking_data = rank_stock_sentiment()
    print("\nTop 10 stocks by sentiment:")
    print(ranking_data.head(10))
    print("\nBottom 10 stocks by sentiment:")
    print(ranking_data.tail(10))
