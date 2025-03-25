import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import zscore

@dataclass
class FactorConfig:
    """Configuration for factor calculation"""
    window_length: int
    universe_size: int = 500  # Top N stocks by volume
    sector_neutral: bool = True
    smoothing_window: Optional[int] = None

class AlphaFactors:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize AlphaFactors with historical data
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary of DataFrames containing historical data for each ticker
            Each DataFrame should have columns: ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data
        self.tickers = list(data.keys())
        
    def calculate_momentum(self, config: FactorConfig) -> pd.DataFrame:
        """
        Calculate momentum factor based on past returns
        
        Parameters
        ----------
        config : FactorConfig
            Configuration for factor calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with momentum factor values for each ticker and date
        """
        momentum_data = {}
        
        for ticker, df in self.data.items():
            # Calculate returns
            returns = df['close'].pct_change()
            
            # Calculate momentum (rolling mean of returns)
            momentum = returns.rolling(window=config.window_length).mean()
            
            # Apply sector neutralization if configured
            if config.sector_neutral:
                momentum = self._sector_neutralize(momentum)
            
            # Apply smoothing if configured
            if config.smoothing_window:
                momentum = momentum.rolling(window=config.smoothing_window).mean()
            
            momentum_data[ticker] = momentum
        
        # Combine into DataFrame and standardize
        momentum_df = pd.DataFrame(momentum_data)
        return self._standardize(momentum_df)
    
    def calculate_mean_reversion(self, config: FactorConfig) -> pd.DataFrame:
        """
        Calculate mean reversion factor based on price deviation from moving average
        
        Parameters
        ----------
        config : FactorConfig
            Configuration for factor calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with mean reversion factor values for each ticker and date
        """
        mean_rev_data = {}
        
        for ticker, df in self.data.items():
            # Calculate moving average
            ma = df['close'].rolling(window=config.window_length).mean()
            
            # Calculate deviation from moving average
            deviation = (df['close'] - ma) / ma
            
            # Apply sector neutralization if configured
            if config.sector_neutral:
                deviation = self._sector_neutralize(deviation)
            
            # Apply smoothing if configured
            if config.smoothing_window:
                deviation = deviation.rolling(window=config.smoothing_window).mean()
            
            mean_rev_data[ticker] = -deviation  # Negative for mean reversion
        
        # Combine into DataFrame and standardize
        mean_rev_df = pd.DataFrame(mean_rev_data)
        return self._standardize(mean_rev_df)
    
    def calculate_overnight_sentiment(self, config: FactorConfig) -> pd.DataFrame:
        """
        Calculate overnight sentiment factor based on overnight returns
        
        Parameters
        ----------
        config : FactorConfig
            Configuration for factor calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with overnight sentiment factor values for each ticker and date
        """
        sentiment_data = {}
        
        for ticker, df in self.data.items():
            # Calculate overnight returns
            overnight_returns = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            
            # Calculate trailing sum of overnight returns
            sentiment = overnight_returns.rolling(window=config.window_length).sum()
            
            # Apply sector neutralization if configured
            if config.sector_neutral:
                sentiment = self._sector_neutralize(sentiment)
            
            # Apply smoothing if configured
            if config.smoothing_window:
                sentiment = sentiment.rolling(window=config.smoothing_window).mean()
            
            sentiment_data[ticker] = sentiment
        
        # Combine into DataFrame and standardize
        sentiment_df = pd.DataFrame(sentiment_data)
        return self._standardize(sentiment_df)
    
    def calculate_volume_trend(self, config: FactorConfig) -> pd.DataFrame:
        """
        Calculate volume trend factor based on volume relative to moving average
        
        Parameters
        ----------
        config : FactorConfig
            Configuration for factor calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with volume trend factor values for each ticker and date
        """
        volume_data = {}
        
        for ticker, df in self.data.items():
            # Calculate volume moving average
            volume_ma = df['volume'].rolling(window=config.window_length).mean()
            
            # Calculate volume trend
            volume_trend = df['volume'] / volume_ma - 1
            
            # Apply sector neutralization if configured
            if config.sector_neutral:
                volume_trend = self._sector_neutralize(volume_trend)
            
            # Apply smoothing if configured
            if config.smoothing_window:
                volume_trend = volume_trend.rolling(window=config.smoothing_window).mean()
            
            volume_data[ticker] = volume_trend
        
        # Combine into DataFrame and standardize
        volume_df = pd.DataFrame(volume_data)
        return self._standardize(volume_df)
    
    def _sector_neutralize(self, factor: pd.Series) -> pd.Series:
        """
        Neutralize factor values by sector
        
        Parameters
        ----------
        factor : pd.Series
            Factor values to neutralize
            
        Returns
        -------
        pd.Series
            Sector-neutralized factor values
        """
        # TODO: Implement sector neutralization once we have sector data
        # For now, just return the original factor
        return factor
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize factor values using z-score
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with factor values
            
        Returns
        -------
        pd.DataFrame
            Standardized factor values
        """
        return df.apply(zscore, axis=1)
    
    def calculate_all_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate all alpha factors with default configurations
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all calculated factors
        """
        # Define configurations for each factor
        momentum_config = FactorConfig(window_length=252, smoothing_window=5)
        mean_rev_config = FactorConfig(window_length=5, smoothing_window=3)
        sentiment_config = FactorConfig(window_length=5, smoothing_window=3)
        volume_config = FactorConfig(window_length=20, smoothing_window=5)
        
        # Calculate all factors
        factors = {
            'momentum': self.calculate_momentum(momentum_config),
            'mean_reversion': self.calculate_mean_reversion(mean_rev_config),
            'overnight_sentiment': self.calculate_overnight_sentiment(sentiment_config),
            'volume_trend': self.calculate_volume_trend(volume_config)
        }
        
        return factors
    
    def combine_factors(self, factors: Dict[str, pd.DataFrame], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Combine multiple factors into a single alpha signal
        
        Parameters
        ----------
        factors : Dict[str, pd.DataFrame]
            Dictionary of factor DataFrames
        weights : Optional[Dict[str, float]]
            Dictionary of weights for each factor. If None, equal weights are used.
            
        Returns
        -------
        pd.DataFrame
            Combined alpha signal
        """
        if weights is None:
            weights = {factor: 1.0/len(factors) for factor in factors.keys()}
        
        # Ensure weights sum to 1
        weight_sum = sum(weights.values())
        weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Combine factors
        combined = pd.DataFrame(0, index=factors[list(factors.keys())[0]].index, 
                              columns=factors[list(factors.keys())[0]].columns)
        
        for factor, df in factors.items():
            combined += df * weights[factor]
        
        return combined

def load_factor_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load historical data for factor calculation
    
    Parameters
    ----------
    data_dir : str
        Directory containing historical data files
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames containing historical data for each ticker
    """
    import glob
    import os
    
    data = {}
    files = glob.glob(os.path.join(data_dir, 'backtest_data_*.csv'))
    
    for file in files:
        ticker = file.split('_')[2]
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        data[ticker] = df
    
    return data

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    data = load_factor_data(data_dir)
    
    # Initialize AlphaFactors
    alpha_factors = AlphaFactors(data)
    
    # Calculate all factors
    factors = alpha_factors.calculate_all_factors()
    
    # Combine factors with custom weights
    weights = {
        'momentum': 0.3,
        'mean_reversion': 0.2,
        'overnight_sentiment': 0.3,
        'volume_trend': 0.2
    }
    
    combined_alpha = alpha_factors.combine_factors(factors, weights)
    
    # Print summary statistics
    print("\nFactor Summary Statistics:")
    for factor, df in factors.items():
        print(f"\n{factor}:")
        print(df.describe())
    
    print("\nCombined Alpha Summary Statistics:")
    print(combined_alpha.describe()) 