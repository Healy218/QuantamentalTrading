import requests
import matplotlib as plt
import yfinance as yf
import cloudscraper
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import re

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
scraper = cloudscraper.create_scraper()
import requests

def get_trending_tickers():
    url = "https://api.stocktwits.com/api/2/streams/trending.json"
    response = scraper.get(url)
    
    # Check the status code first
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        print("Response text:", response.text)
        return []  # Return an empty list or handle as needed

    # Try decoding JSON
    try:
        data = response.json()
    except Exception as e:
        print("Failed to decode JSON:", e)
        print("Response content:", response.text)
        return []
    
    # Parse tickers from the messages
    tickers = [message['symbols'][0]['symbol'] for message in data.get('messages', []) if message.get('symbols')]
    unique_tickers = list(set(tickers))[:10]
    print("Top 10 Trending Tickers:", unique_tickers)
    return unique_tickers

# def get_most_active_tickers():
#     """
#     Scrape the most active tickers from StockTwits sentiment page
#     Returns a list of dictionaries containing ticker and percentage change
#     """
#     url = "https://stocktwits.com/sentiment/most-active"
#     response = scraper.get(url)
    
#     if response.status_code != 200:
#         print(f"Error fetching data: {response.status_code}")
#         print("Response text:", response.text)
#         return []

#     try:
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # Find the main content area containing the tickers
#         main_content = soup.find('main')
#         if not main_content:
#             print("Could not find main content area")
#             return []
            
#         # Find all ticker elements
#         ticker_elements = main_content.find_all('div', class_=lambda x: x and 'ticker' in x.lower())
        
#         active_tickers = []
#         for element in ticker_elements:
#             # Get the ticker symbol
#             ticker = element.find('span', class_=lambda x: x and 'symbol' in x.lower())
#             # Get the percentage change
#             percentage = element.find('span', class_=lambda x: x and 'change' in x.lower())
            
#             if ticker and percentage:
#                 active_tickers.append({
#                     'ticker': ticker.text.strip(),
#                     'percentage_change': percentage.text.strip()
#                 })
        
#         # Get top 10 most active tickers
#         top_10 = active_tickers[:10]
#         print("\nTop 10 Most Active Tickers:")
#         for ticker_data in top_10:
#             print(f"{ticker_data['ticker']}: {ticker_data['percentage_change']}")
        
#         return top_10
        
#     except Exception as e:
#         print(f"Error parsing data: {e}")
#         print("Response content:", response.text[:500])  # Print first 500 chars of response for debugging
#         return []

if __name__ == "__main__":
    print("Fetching trending tickers from StockTwits...")
    trending_tickers = get_trending_tickers()
    
    # print("\nFetching most active tickers from StockTwits sentiment page...")
    # active_tickers = get_most_active_tickers()