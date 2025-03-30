import pandas as pd
import numpy as np
import yfinance as yf
import talib
import backtrader as bt
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from strategies.momenturmburst import MomentumBurstStrategy
from strategies.orb import OpeningRangeBreakoutStrategy
from strategies.rsidivergence import RSIDivergenceStrategy
from strategies.volumebreakout import VolumeBreakoutStrategy
from strategies.macdzerocross import MACDZeroCrossStrategy

# Custom VWAP Indicator
class VWAP(bt.Indicator):
    lines = ('vwap',)
    params = (('period', None),)

    def __init__(self):
        self.addminperiod(1)
        self.cum_volume = 0.0
        self.cum_price_volume = 0.0

    def next(self):
        price = (self.data.high + self.data.low + self.data.close) / 3
        volume = self.data.volume
        self.cum_price_volume += price * volume
        self.cum_volume += volume
        self.lines.vwap[0] = self.cum_price_volume / self.cum_volume if self.cum_volume > 0 else 0

# ReversalAtKeyLevelsStrategy with Custom VWAP
class ReversalAtKeyLevelsStrategy(bt.Strategy):
    def __init__(self):
        self.vwap = VWAP(self.data)
        self.order = None
        self.trade_state = 0
        self.portfolio_values = []
        self.order_count = 0

    def next(self):
        self.portfolio_values.append(self.broker.getvalue())
        if self.order:
            return
        body = abs(self.data.close[0] - self.data.open[0])
        lower_wick = min(self.data.close[0], self.data.open[0]) - self.data.low[0]
        is_hammer = (lower_wick > 2 * body) and (self.data.close[0] > self.vwap[0])
        is_doji = (body < 0.0001 * self.data.close[0]) and (self.data.close[0] < self.vwap[0])
        if not self.position:
            if is_hammer:
                self.order = self.buy()
                self.trade_state = 1
                self.order_count += 1
            elif is_doji:
                self.order = self.sell()
                self.trade_state = -1
                self.order_count += 1
        elif self.trade_state == 1:
            self.order = self.close()
            self.trade_state = 0
            self.order_count += 1
        elif self.trade_state == -1:
            self.order = self.close()
            self.trade_state = 0
            self.order_count += 1

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            pass
        self.order = None

# Custom PandasData class
class CustomPandasData(bt.feeds.PandasData):
    lines = ('Signal',)
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),      
        ('volume', 'Volume'),
        ('openinterest', None),
        ('Signal', 'Signal'),
    )

# Data Acquisition
def get_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if data.empty:
        print(f"Warning: No data downloaded for {ticker}")
        return None
    data.columns = [col[0] for col in data.columns]
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_columns):
        print(f"Warning: Missing columns for {ticker}: {data.columns}")
        return None
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')
    return data

# Feature Engineering
def calculate_indicators(df):
    close_array = df['Close'].values.flatten()
    df['EMA_20'] = talib.EMA(close_array, timeperiod=20)
    df['RSI'] = talib.RSI(close_array, timeperiod=14)
    macd, signal, _ = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal'] = signal
    upper, middle, lower = talib.BBANDS(close_array, timeperiod=20)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    return df

# Signal Generation
def generate_signals(df):
    df['Signal'] = 0
    buy_condition_rsi = (df['RSI'].shift(1) <= 45) & (df['RSI'] > 45)
    buy_condition_macd = (df['MACD'].shift(1) <= df['Signal'].shift(1)) & (df['MACD'] > df['Signal'])
    df.loc[buy_condition_rsi | buy_condition_macd, 'Signal'] = 1
    sell_condition_rsi = (df['RSI'].shift(1) >= 65) & (df['RSI'] < 65)
    sell_condition_macd = (df['MACD'].shift(1) >= df['Signal'].shift(1)) & (df['MACD'] < df['Signal'])
    df.loc[sell_condition_rsi | sell_condition_macd, 'Signal'] = -1
    return df

# Original Strategy
class OptionStrategy(bt.Strategy):
    def __init__(self):
        self.signal = self.datas[0].Signal
        self.portfolio_values = []
        self.trade_state = 0
        self.order_count = 0

    def next(self):
        self.portfolio_values.append(self.broker.getvalue())
        if not self.trade_state:
            if self.signal[0] == 1:
                self.buy()
                self.trade_state = 1
                self.order_count += 1
            elif self.signal[0] == -1:
                self.sell()
                self.trade_state = -1
                self.order_count += 1
        elif self.trade_state == 1:
            if self.signal[0] == -1:
                self.close()
                self.trade_state = 0
                self.order_count += 1
        elif self.trade_state == -1:
            if self.signal[0] == 1:
                self.close()
                self.trade_state = 0
                self.order_count += 1

# Neural Network
class StrategySelector(nn.Module):
    def __init__(self, input_size, hidden_size, num_strategies):
        super(StrategySelector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_strategies)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Backtest Function
def run_backtest(data, strategy_class, start_capital=10000.0):
    cerebro = bt.Cerebro()
    data_feed = CustomPandasData(dataname=data)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(start_capital)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    try:
        strategies = cerebro.run()
        strategy = strategies[0]
        final_value = cerebro.broker.getvalue()
        order_count = strategy.order_count if hasattr(strategy, 'order_count') else 0
        return final_value, strategy.portfolio_values if hasattr(strategy, 'portfolio_values') else [final_value] * len(data), order_count
    except Exception as e:
        print(f"Error in strategy {strategy_class.__name__}: {str(e)}")
        return start_capital, [start_capital] * len(data), 0

# Optimization Function
def optimize_strategies(tickers, start_date, end_date, interval):
    strategies = {
        "original": OptionStrategy,
        "momentum": MomentumBurstStrategy,
        "orb": OpeningRangeBreakoutStrategy,
        "macd": MACDZeroCrossStrategy,
        "reversal": ReversalAtKeyLevelsStrategy,
        "rsi": RSIDivergenceStrategy,
        "volume": VolumeBreakoutStrategy
    }
    
    start_capital = 10000.0
    results = {}
    total_iterations = len(tickers) * len(strategies)
    
    # Backtesting with progress bar
    with tqdm(total=total_iterations, desc="Backtesting Progress") as pbar:
        for ticker in tickers:
            try:
                data = get_data(ticker, start_date, end_date, interval)
                if data is None:
                    pbar.update(len(strategies))
                    continue
                data = calculate_indicators(data.copy())
                data = generate_signals(data.copy())
                results[ticker] = {"strategies": {}, "features": None}
                features = [
                    np.mean(data['Close']),
                    np.std(data['Close']),
                    data['Volume'].mean()
                ]
                results[ticker]["features"] = features
                for strat_name, strat_class in strategies.items():
                    final_value, portfolio_values, order_count = run_backtest(data, strat_class, start_capital)
                    returns = (final_value - start_capital) / start_capital
                    results[ticker]["strategies"][strat_name] = {
                        "returns": returns,
                        "final_value": final_value,
                        "order_count": order_count
                    }
                    pbar.update(1)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                pbar.update(len(strategies))

    if not results:
        print("No valid results to process. Exiting.")
        return {}, []

    # Prepare data for PyTorch
    num_strategies = len(strategies)
    strategy_names = list(strategies.keys())
    input_data = []
    target_data = []
    
    for ticker in results:
        returns_dict = results[ticker]["strategies"]
        returns = [returns_dict.get(strat, {"returns": 0})["returns"] for strat in strategy_names]
        input_features = results[ticker]["features"]
        input_data.append(input_features)
        best_strategy = strategy_names[np.argmax(returns)]
        target_data.append(best_strategy)

    X = torch.tensor(input_data, dtype=torch.float32)
    y = torch.tensor([strategy_names.index(name) for name in target_data], dtype=torch.long)

    input_size = len(input_data[0])
    hidden_size = 10
    model = StrategySelector(input_size, hidden_size, num_strategies)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 100
    with tqdm(total=epochs, desc="Training Neural Network") as pbar:
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            if (epoch + 1) % 10 == 0:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    with torch.no_grad():
        predictions = model(X)
        predicted_indices = torch.argmax(predictions, dim=1)
        best_strategies = [strategy_names[idx] for idx in predicted_indices]

    # Enhanced output
    print("\n=== Strategy Results ===")
    for ticker, best_strat in zip(tickers, best_strategies):
        if ticker not in results or best_strat not in results[ticker]["strategies"]:
            continue
        best_data = results[ticker]["strategies"][best_strat]
        print(f"\nBest Strategy for {ticker}: {best_strat}")
        print(f"  Returns: {best_data['returns']:.2%}")
        print(f"  Final Portfolio Value: ${best_data['final_value']:.2f}")
        print(f"  Number of Orders: {best_data['order_count']}")

        # Find worst strategy
        returns = [results[ticker]["strategies"][strat]["returns"] for strat in strategy_names if strat in results[ticker]["strategies"]]
        if returns:
            worst_strat = strategy_names[np.argmin(returns)]
            worst_data = results[ticker]["strategies"][worst_strat]
            print(f"\nWorst Strategy for {ticker}: {worst_strat}")
            print(f"  Returns: {worst_data['returns']:.2%}")
            print(f"  Final Portfolio Value: ${worst_data['final_value']:.2f}")
            print(f"  Number of Orders: {worst_data['order_count']}")

    return results, best_strategies

if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "VTI", "TSLA", "NVDA", "MSFT", "AMZN", "GOOG", "TSM", "WMT", "NFLX", "ORCL", "JNJ", "JPM", "V", "PG", "MA", "DIS", "TM", "IBM", "MMM", "PFE", "ABBV", "WBA", "BA", "CAT", "CSCO", "CVX", "DD", "GE", "GS", "HD", "IBM", "JNJ", "KO", "LLY", "MCD", "MMM", "MRK", "MS", "NFLX", "ORCL", "PFE", "PG", "TSM", "V", "WMT"]
    start_date = "2025-02-01"
    end_date = "2025-03-28"
    interval = "5m"

    results, best_strategies = optimize_strategies(tickers, start_date, end_date, interval)