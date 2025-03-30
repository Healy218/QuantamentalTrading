def backtest(df, initial_capital=10000):
    df['position'] = df['signal'].shift(1)  # Trade on next bar
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']
    
    # Calculate equity curve
    df['equity'] = initial_capital * (1 + df['strategy_returns']).cumprod()
    
    # Metrics
    total_return = (df['equity'].iloc[-1] / initial_capital) - 1
    sharpe_ratio = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252 * 390)  # 390 min/day
    max_drawdown = (df['equity'] / df['equity'].cummax() - 1).min()
    
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Equity Curve": df['equity']
    }

# Run backtest
results = backtest(df)
print(f"Total Return: {results['Total Return']:.2%}")
print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
print(f"Max Drawdown: {results['Max Drawdown']:.2%}")

# Plot equity curve
import matplotlib.pyplot as plt
plt.plot(df.index, results['Equity Curve'], label="Strategy Equity")
plt.title("Momentum Burst Backtest on SPY")
plt.xlabel("Time")
plt.ylabel("Equity ($)")
plt.legend()
plt.show()