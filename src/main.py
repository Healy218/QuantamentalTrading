import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# Import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis/quantitative'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis/sentiment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'analysis/fundamental'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backtesting'))

import AlphaFactors
import StockTwitsSentiment
import TweetsRNN
from Backtests import BacktestEngine

class QuantamentalTrading:
    def __init__(self):
        self.universe = []  # List of stock tickers
        self.alpha_signals = {}  # Dictionary of alpha signals for each stock
        self.fundamental_scores = {}  # Dictionary of fundamental scores
        self.sentiment_scores = {}  # Dictionary of sentiment scores
        self.final_portfolio = {}  # Final portfolio weights
        self.market_data = {}  # Market data for analysis
        
        # Initialize data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Default universe - can be modified
        self.universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    def collect_market_data(self, start_date, end_date):
        """
        Collect market data for analysis
        - Price data
        - Volume data
        - Fundamental data from SEC filings
        - Social media data
        """
        print("Collecting market data...")
        
        # Collect price and volume data using yfinance
        for ticker in self.universe:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                self.market_data[ticker] = {
                    'price': stock_data['Adj Close'],
                    'volume': stock_data['Volume'],
                    'returns': stock_data['Adj Close'].pct_change()
                }
            except Exception as e:
                print(f"Error collecting data for {ticker}: {e}")
                
        # Save market data for later use
        market_data_path = self.data_dir / "market_data.pkl"
        pd.to_pickle(self.market_data, market_data_path)
        
        return self.market_data

    def generate_alpha_signals(self):
        """
        Generate alpha signals using quantitative factors:
        - Momentum factors
        - Value factors
        - Quality factors
        - etc.
        """
        print("Generating alpha signals...")
        
        # Initialize alpha factors
        alpha_engine = AlphaFactors.AlphaEngine(self.market_data)
        
        # Generate various alpha signals
        for ticker in self.universe:
            if ticker in self.market_data:
                price_data = self.market_data[ticker]['price']
                volume_data = self.market_data[ticker]['volume']
                
                self.alpha_signals[ticker] = {
                    'momentum': alpha_engine.calculate_momentum(price_data),
                    'volume_signal': alpha_engine.calculate_volume_signal(volume_data),
                    'mean_reversion': alpha_engine.calculate_mean_reversion(price_data)
                }
        
        return self.alpha_signals

    def analyze_fundamentals(self):
        """
        Analyze fundamental data:
        - 10-K reports
        - Quarterly reports
        - Financial ratios
        """
        print("Analyzing fundamental data...")
        
        # Use EDGAR tools to analyze fundamentals
        edgar_dir = self.data_dir / "edgar"
        edgar_dir.mkdir(exist_ok=True)
        
        for ticker in self.universe:
            try:
                # Download and analyze latest 10-K
                edgar_data = self._download_latest_10k(ticker)
                
                # Calculate fundamental scores
                self.fundamental_scores[ticker] = {
                    'profitability': self._calculate_profitability_score(edgar_data),
                    'growth': self._calculate_growth_score(edgar_data),
                    'value': self._calculate_value_score(edgar_data)
                }
            except Exception as e:
                print(f"Error analyzing fundamentals for {ticker}: {e}")
        
        return self.fundamental_scores

    def analyze_sentiment(self):
        """
        Analyze sentiment from social media:
        - StockTwits data
        - Twitter data
        """
        print("Analyzing sentiment data...")
        
        # Initialize sentiment analyzers
        stocktwits_analyzer = StockTwitsSentiment.SentimentAnalyzer()
        twitter_analyzer = TweetsRNN.TwitterSentimentAnalyzer()
        
        for ticker in self.universe:
            try:
                # Analyze StockTwits sentiment
                stocktwits_sentiment = stocktwits_analyzer.analyze_sentiment(ticker)
                
                # Analyze Twitter sentiment
                twitter_sentiment = twitter_analyzer.analyze_sentiment(ticker)
                
                # Combine sentiment scores
                self.sentiment_scores[ticker] = {
                    'stocktwits': stocktwits_sentiment,
                    'twitter': twitter_sentiment,
                    'combined': (stocktwits_sentiment + twitter_sentiment) / 2
                }
            except Exception as e:
                print(f"Error analyzing sentiment for {ticker}: {e}")
        
        return self.sentiment_scores

    def combine_signals(self):
        """
        Combine all signals to generate final portfolio:
        - Weight different signals
        - Apply risk constraints
        - Generate final positions
        """
        print("Combining signals for final portfolio...")
        
        # Weights for different signal types
        weights = {
            'alpha': 0.4,
            'fundamental': 0.4,
            'sentiment': 0.2
        }
        
        for ticker in self.universe:
            try:
                # Normalize signals
                alpha_score = self._normalize_alpha_signals(self.alpha_signals[ticker])
                fundamental_score = self._normalize_fundamental_signals(self.fundamental_scores[ticker])
                sentiment_score = self.sentiment_scores[ticker]['combined']
                
                # Calculate combined score
                combined_score = (
                    weights['alpha'] * alpha_score +
                    weights['fundamental'] * fundamental_score +
                    weights['sentiment'] * sentiment_score
                )
                
                self.final_portfolio[ticker] = combined_score
                
            except Exception as e:
                print(f"Error combining signals for {ticker}: {e}")
        
        # Normalize portfolio weights to sum to 1
        total_score = sum(self.final_portfolio.values())
        self.final_portfolio = {
            ticker: score/total_score 
            for ticker, score in self.final_portfolio.items()
        }
        
        return self.final_portfolio

    def backtest_strategy(self):
        """
        Backtest the combined strategy:
        - Calculate returns
        - Analyze performance metrics
        - Generate reports
        """
        print("Backtesting strategy...")
        
        # Initialize backtest engine
        backtest = BacktestEngine(
            portfolio_weights=self.final_portfolio,
            market_data=self.market_data
        )
        
        # Run backtest
        results = backtest.run()
        
        # Generate performance metrics
        metrics = backtest.calculate_metrics()
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        return results, metrics

    def _normalize_alpha_signals(self, alpha_signals):
        """Helper function to normalize alpha signals"""
        signals = np.array(list(alpha_signals.values()))
        return np.mean((signals - np.mean(signals)) / np.std(signals))

    def _normalize_fundamental_signals(self, fundamental_signals):
        """Helper function to normalize fundamental signals"""
        signals = np.array(list(fundamental_signals.values()))
        return np.mean((signals - np.mean(signals)) / np.std(signals))

    def _download_latest_10k(self, ticker):
        """Helper function to download latest 10-K report"""
        # Placeholder - implement actual 10-K download logic
        return {}

    def _calculate_profitability_score(self, edgar_data):
        """Helper function to calculate profitability score"""
        # Placeholder - implement actual profitability calculation
        return np.random.random()

    def _calculate_growth_score(self, edgar_data):
        """Helper function to calculate growth score"""
        # Placeholder - implement actual growth calculation
        return np.random.random()

    def _calculate_value_score(self, edgar_data):
        """Helper function to calculate value score"""
        # Placeholder - implement actual value calculation
        return np.random.random()

    def run_pipeline(self, start_date, end_date):
        """
        Run the complete quantamental trading pipeline
        """
        print(f"Starting quantamental trading pipeline for {start_date} to {end_date}")
        
        self.collect_market_data(start_date, end_date)
        self.generate_alpha_signals()
        self.analyze_fundamentals()
        self.analyze_sentiment()
        self.combine_signals()
        results, metrics = self.backtest_strategy()
        
        print("\nPipeline completed!")
        return results, metrics

def main():
    # Example usage
    trader = QuantamentalTrading()
    
    # Set date range for analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    # Run the complete pipeline
    results, metrics = trader.run_pipeline(start_date, end_date)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    pd.to_pickle(results, results_dir / "backtest_results.pkl")
    pd.to_pickle(metrics, results_dir / "performance_metrics.pkl")

if __name__ == "__main__":
    main() 