# QuantamentalTrading

A comprehensive quantamental trading system that combines quantitative analysis, fundamental analysis, and sentiment analysis for making investment decisions.

## Project Structure

```
QuantamentalTrading/
├── src/
│   ├── data_collection/      # Data collection modules for market, fundamental, and social data
│   ├── analysis/
│   │   ├── quantitative/     # Quantitative analysis and alpha factor generation
│   │   ├── fundamental/      # Fundamental analysis using SEC filings
│   │   └── sentiment/        # Sentiment analysis for social media data
│   ├── backtesting/         # Backtesting framework and performance analysis
│   ├── utils/               # Utility functions and helpers
│   └── main.py             # Main entry point for the trading system
├── tests/                  # Unit tests and integration tests
└── docs/                   # Documentation and strategy details

```

## Components

1. **Quantitative Analysis**

   - Alpha factor generation
   - Risk factor modeling
   - Portfolio optimization

2. **Fundamental Analysis**

   - SEC filing analysis (10-K, 10-Q)
   - Financial ratio analysis
   - Company fundamentals

3. **Sentiment Analysis**

   - StockTwits sentiment analysis
   - Twitter sentiment analysis using RNNs
   - Social media trend analysis

4. **Backtesting**
   - Historical performance analysis
   - Risk metrics calculation
   - Transaction cost analysis

## Setup and Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure data sources:

   - Set up API keys for market data
   - Configure SEC EDGAR access
   - Set up social media API access

3. Run the system:
   ```bash
   python src/main.py
   ```

## Trading Process

1. **Data Collection**

   - Market data (prices, volumes)
   - Fundamental data (financial statements)
   - Social media data

2. **Signal Generation**

   - Generate alpha factors
   - Analyze fundamental metrics
   - Calculate sentiment scores

3. **Portfolio Construction**

   - Combine multiple signals
   - Apply risk constraints
   - Generate final positions

4. **Performance Analysis**
   - Backtest strategy
   - Calculate performance metrics
   - Generate reports

## Contributing

Feel free to contribute by:

1. Opening issues for bugs or feature requests
2. Submitting pull requests
3. Improving documentation

## License

[Your chosen license]

StocktwitsSentiment function take in a JSON file from a TwitsFileMaker and trains a model to preform Sentiment analysis on StockTwits.

TweetsRNN takes in a JSON file from TweetsFileMaker and trains a model to preform Sentiment analysis on Tweets from a list of twitter accounts.

Vocab folder creates a vocabulary of weighted works for use in the RNNs.
