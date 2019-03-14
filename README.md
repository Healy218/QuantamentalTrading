# QuantamentalTrading

Qantamental Trading is the symbolic union of Fundamental Analysis with Quanatative Analysis for use in economic Markets.

This repository will have several statistical analysis' written in python that looks at past stock data and finds statistical similarities towards mispricing (Quanatative Analysis), as well as Sentiment RNNs for stock tickers using different social media platforms (Fundamental Analysis).

As time goes on there will be more modules added that combine Fundamental and Quantatative Analysis to hopefully better predict mispricing in stocks and better predict future returns.

StocktwitsSentiment function take in a JSON file from a TwitsFileMaker and trains a model to preform Sentiment analysis on StockTwits. 

TweetsRNN takes in a JSON file from TweetsFileMaker and trains a model to preform Sentiment analysis on Tweets from a list of twitter accounts. 

Vocab folder creates a vocabulary of weighted works for use in the RNNs. 

