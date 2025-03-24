import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_alpha import load_backtest_data, calculate_returns, calculate_technical_indicators, backtest_strategy, calculate_performance_metrics

def run_dashboard():
    st.title("Trading Strategy Backtest Dashboard")
    
    # Load data
    with st.spinner("Loading backtest data..."):
        historical_data = load_backtest_data()
        tickers = list(historical_data.keys())
    
    # Sidebar controls
    st.sidebar.header("Controls")
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)
    
    # Run backtest for selected ticker
    if selected_ticker:
        df = historical_data[selected_ticker]
        df = calculate_returns(df)
        df = calculate_technical_indicators(df)
        results = backtest_strategy(df)
        metrics = calculate_performance_metrics(results)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col2:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        with col3:
            st.metric("Avg Position Size", f"{metrics['avg_position_size']:.2f}%")
            positions_taken = (results['position'] > 0).sum()
            st.metric("Positions Taken", positions_taken)
        
        # Signal Analysis
        st.subheader("Signal Analysis")
        signal_col1, signal_col2 = st.columns(2)
        with signal_col1:
            st.write("Signal Counts:")
            st.write(f"- Momentum: {results['momentum_signal'].sum()}")
            st.write(f"- Value: {results['value_signal'].sum()}")
            st.write(f"- RSI: {results['RSI_signal'].sum()}")
            st.write(f"- MACD: {results['MACD_signal'].sum()}")
        with signal_col2:
            st.write("Signal Strength:")
            st.write(f"- Average: {results['signal_strength'].mean():.2f}")
            st.write(f"- Strong Signals (>=2): {(results['signal_strength'] >= 2).sum()}")
        
        # Interactive plots
        st.subheader("Price and Indicators")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(results.index, results['close'], label='Price', alpha=0.7)
        ax1.plot(results.index, results['SMA_8'], label='8-day SMA', alpha=0.7)
        ax1.plot(results.index, results['SMA_40'], label='40-day SMA', alpha=0.7)
        ax1.set_title(f'{selected_ticker} Price and Moving Averages')
        ax1.legend()
        st.pyplot(fig1)
        
        st.subheader("Technical Indicators")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(results.index, results['RSI'], label='RSI', alpha=0.7)
        ax2.plot(results.index, results['MACD'], label='MACD', alpha=0.7)
        ax2.plot(results.index, results['Signal_Line'], label='Signal Line', alpha=0.7)
        ax2.set_title('Technical Indicators')
        ax2.legend()
        st.pyplot(fig2)
        
        st.subheader("Portfolio Performance")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(results.index, results['portfolio_value'], label='Portfolio Value', alpha=0.7)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(results.index, results['position'], label='Position Size', color='orange', alpha=0.7)
        ax3.set_title('Portfolio Value and Position Size')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        st.pyplot(fig3)
        
        # Display raw data
        if st.checkbox("Show Raw Data"):
            st.dataframe(results)

if __name__ == "__main__":
    run_dashboard() 