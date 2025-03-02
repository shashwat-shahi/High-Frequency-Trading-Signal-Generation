"""
Feature engineering pipeline for high-frequency trading with statistical significance testing.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats
from scipy.stats import jarque_bera, normaltest, kstest
import warnings
from collections import deque
from dataclasses import dataclass
import time
from numba import jit
from ..data.orderbook import OrderBook, Tick


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    lookback_windows: List[int] = None
    price_features: bool = True
    volume_features: bool = True
    microstructure_features: bool = True
    technical_features: bool = True
    statistical_features: bool = True
    significance_level: float = 0.05
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 50, 100]


class StatisticalTester:
    """Statistical significance testing for features."""
    
    @staticmethod
    def test_stationarity(series: np.ndarray, significance_level: float = 0.05) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        from statsmodels.tsa.stattools import adfuller
        
        try:
            result = adfuller(series, autolag='AIC')
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < significance_level
            }
        except Exception as e:
            return {
                'adf_statistic': np.nan,
                'p_value': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'error': str(e)
            }
    
    @staticmethod
    def test_normality(series: np.ndarray, significance_level: float = 0.05) -> Dict[str, Any]:
        """Test for normality using multiple tests."""
        results = {}
        
        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(series)
            results['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > significance_level
            }
        except Exception:
            results['jarque_bera'] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
        
        # D'Agostino's normality test
        try:
            da_stat, da_p = normaltest(series)
            results['dagostino'] = {
                'statistic': da_stat,
                'p_value': da_p,
                'is_normal': da_p > significance_level
            }
        except Exception:
            results['dagostino'] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
        
        return results
    
    @staticmethod
    def test_autocorrelation(series: np.ndarray, max_lags: int = 20, 
                           significance_level: float = 0.05) -> Dict[str, Any]:
        """Test for autocorrelation using Ljung-Box test."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            result = acorr_ljungbox(series, lags=max_lags, return_df=True)
            significant_lags = result[result['lb_pvalue'] < significance_level].index.tolist()
            
            return {
                'ljung_box_statistics': result['lb_stat'].to_dict(),
                'p_values': result['lb_pvalue'].to_dict(),
                'significant_lags': significant_lags,
                'has_autocorrelation': len(significant_lags) > 0
            }
        except Exception as e:
            return {
                'ljung_box_statistics': {},
                'p_values': {},
                'significant_lags': [],
                'has_autocorrelation': False,
                'error': str(e)
            }


@jit(nopython=True)
def rolling_mean_numba(x: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean calculation using numba."""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.mean(x[i - window + 1:i + 1])
    
    return result


@jit(nopython=True)
def rolling_std_numba(x: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling standard deviation calculation using numba."""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        result[i] = np.std(x[i - window + 1:i + 1])
    
    return result


@jit(nopython=True)
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate log returns."""
    n = len(prices)
    returns = np.full(n, np.nan)
    
    for i in range(1, n):
        if prices[i] > 0 and prices[i-1] > 0:
            returns[i] = np.log(prices[i] / prices[i-1])
    
    return returns


class FeatureEngine:
    """
    High-performance feature engineering pipeline for HFT with statistical validation.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.tester = StatisticalTester()
        self.feature_cache = {}
        self.feature_stats = {}
        
    def extract_price_features(self, order_book: OrderBook, prices: np.ndarray) -> Dict[str, float]:
        """Extract price-based features."""
        features = {}
        
        if len(prices) < 2:
            return features
            
        # Current price metrics
        current_price = prices[-1] if len(prices) > 0 else np.nan
        features['current_price'] = current_price
        features['mid_price'] = order_book.get_mid_price() or np.nan
        features['weighted_mid_price'] = order_book.get_weighted_mid_price() or np.nan
        features['spread'] = order_book.get_spread() or np.nan
        
        # Relative spread
        mid_price = features['mid_price']
        if not np.isnan(mid_price) and mid_price > 0 and not np.isnan(features['spread']):
            features['relative_spread'] = features['spread'] / mid_price
        else:
            features['relative_spread'] = np.nan
        
        # Price movements
        returns = calculate_returns(prices)
        features['current_return'] = returns[-1] if len(returns) > 0 else np.nan
        
        # Rolling statistics for different windows
        for window in self.config.lookback_windows:
            if len(prices) >= window:
                window_prices = prices[-window:]
                window_returns = returns[-window:]
                
                # Price statistics
                features[f'price_mean_{window}'] = np.mean(window_prices)
                features[f'price_std_{window}'] = np.std(window_prices)
                features[f'price_min_{window}'] = np.min(window_prices)
                features[f'price_max_{window}'] = np.max(window_prices)
                
                # Return statistics
                valid_returns = window_returns[~np.isnan(window_returns)]
                if len(valid_returns) > 1:
                    features[f'return_mean_{window}'] = np.mean(valid_returns)
                    features[f'return_std_{window}'] = np.std(valid_returns)
                    features[f'return_skew_{window}'] = stats.skew(valid_returns)
                    features[f'return_kurt_{window}'] = stats.kurtosis(valid_returns)
                
                # Price momentum
                if len(window_prices) >= 2:
                    momentum = (window_prices[-1] - window_prices[0]) / window_prices[0]
                    features[f'momentum_{window}'] = momentum
        
        return features
    
    def extract_volume_features(self, order_book: OrderBook, volumes: np.ndarray) -> Dict[str, float]:
        """Extract volume-based features."""
        features = {}
        
        if len(volumes) == 0:
            return features
            
        # Current volume metrics
        features['current_volume'] = volumes[-1] if len(volumes) > 0 else np.nan
        
        # Order book volume features
        features['bid_volume_level_0'] = order_book.get_volume_at_level(0, 'bid')
        features['ask_volume_level_0'] = order_book.get_volume_at_level(0, 'ask')
        
        # Volume at multiple levels
        total_bid_volume = sum(order_book.get_volume_at_level(i, 'bid') for i in range(5))
        total_ask_volume = sum(order_book.get_volume_at_level(i, 'ask') for i in range(5))
        features['total_bid_volume_5'] = total_bid_volume
        features['total_ask_volume_5'] = total_ask_volume
        
        # Volume imbalance
        features['volume_imbalance'] = order_book.get_order_book_imbalance()
        
        # Rolling volume statistics
        for window in self.config.lookback_windows:
            if len(volumes) >= window:
                window_volumes = volumes[-window:]
                
                features[f'volume_mean_{window}'] = np.mean(window_volumes)
                features[f'volume_std_{window}'] = np.std(window_volumes)
                features[f'volume_max_{window}'] = np.max(window_volumes)
                features[f'volume_min_{window}'] = np.min(window_volumes)
                
                # Volume weighted average price (VWAP) approximation
                if len(window_volumes) > 0:
                    features[f'volume_ratio_{window}'] = volumes[-1] / np.mean(window_volumes)
        
        return features
    
    def extract_microstructure_features(self, order_book: OrderBook) -> Dict[str, float]:
        """Extract market microstructure features."""
        features = {}
        
        # Order book shape features
        if len(order_book.bids) > 0 and len(order_book.asks) > 0:
            # Price levels
            features['num_bid_levels'] = len(order_book.bids)
            features['num_ask_levels'] = len(order_book.asks)
            
            # Depth features
            if len(order_book.bids) >= 3:
                features['bid_depth_ratio_1_2'] = (order_book.bids[1].quantity / 
                                                  order_book.bids[0].quantity if order_book.bids[0].quantity > 0 else np.nan)
                features['bid_depth_ratio_2_3'] = (order_book.bids[2].quantity / 
                                                  order_book.bids[1].quantity if order_book.bids[1].quantity > 0 else np.nan)
            
            if len(order_book.asks) >= 3:
                features['ask_depth_ratio_1_2'] = (order_book.asks[1].quantity / 
                                                  order_book.asks[0].quantity if order_book.asks[0].quantity > 0 else np.nan)
                features['ask_depth_ratio_2_3'] = (order_book.asks[2].quantity / 
                                                  order_book.asks[1].quantity if order_book.asks[1].quantity > 0 else np.nan)
            
            # Price impact estimation
            total_bid_qty = sum(level.quantity for level in order_book.bids[:5])
            total_ask_qty = sum(level.quantity for level in order_book.asks[:5])
            
            if total_bid_qty > 0 and total_ask_qty > 0:
                features['liquidity_ratio'] = min(total_bid_qty, total_ask_qty) / max(total_bid_qty, total_ask_qty)
            
        # Tick analysis
        if len(order_book.tick_history) >= 2:
            recent_ticks = list(order_book.tick_history)[-10:]  # Last 10 ticks
            
            # Trade direction
            buy_volume = sum(tick.quantity for tick in recent_ticks if tick.side == 'buy')
            sell_volume = sum(tick.quantity for tick in recent_ticks if tick.side == 'sell')
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                features['buy_sell_ratio'] = buy_volume / total_volume
            
            # Trade frequency
            if len(recent_ticks) >= 2:
                time_diffs = [recent_ticks[i].timestamp - recent_ticks[i-1].timestamp 
                             for i in range(1, len(recent_ticks))]
                features['avg_trade_interval'] = np.mean(time_diffs)
                features['trade_frequency'] = 1.0 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
        
        return features
    
    def extract_technical_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Extract technical analysis features."""
        features = {}
        
        if len(prices) < 20:  # Need minimum data for technical indicators
            return features
        
        # Simple moving averages
        for window in [5, 10, 20]:
            if len(prices) >= window:
                sma = rolling_mean_numba(prices, window)
                features[f'sma_{window}'] = sma[-1]
                
                # Price relative to SMA
                if not np.isnan(sma[-1]) and sma[-1] > 0:
                    features[f'price_sma_ratio_{window}'] = prices[-1] / sma[-1]
        
        # Bollinger Bands
        if len(prices) >= 20:
            sma_20 = rolling_mean_numba(prices, 20)
            std_20 = rolling_std_numba(prices, 20)
            
            if not np.isnan(sma_20[-1]) and not np.isnan(std_20[-1]):
                upper_band = sma_20[-1] + 2 * std_20[-1]
                lower_band = sma_20[-1] - 2 * std_20[-1]
                
                features['bb_upper'] = upper_band
                features['bb_lower'] = lower_band
                features['bb_position'] = ((prices[-1] - lower_band) / 
                                         (upper_band - lower_band) if upper_band != lower_band else 0.5)
        
        # RSI approximation
        if len(prices) >= 14:
            returns = calculate_returns(prices)
            recent_returns = returns[-14:]
            positive_returns = recent_returns[recent_returns > 0]
            negative_returns = recent_returns[recent_returns < 0]
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_gain = np.mean(positive_returns)
                avg_loss = -np.mean(negative_returns)
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features['rsi'] = rsi
        
        return features
    
    def validate_features(self, features: Dict[str, float], 
                         feature_history: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Validate features using statistical tests."""
        validation_results = {}
        
        for feature_name, value in features.items():
            if np.isnan(value) or np.isinf(value):
                validation_results[feature_name] = {
                    'is_valid': False,
                    'reason': 'NaN or Inf value'
                }
                continue
            
            # Basic validation
            is_valid = True
            reason = None
            
            # Check for extreme values (simple outlier detection)
            if abs(value) > 1e6:
                is_valid = False
                reason = 'Extreme value'
            
            validation_results[feature_name] = {
                'is_valid': is_valid,
                'reason': reason,
                'value': value
            }
            
            # Statistical tests if historical data is available
            if feature_history is not None and feature_name in feature_history.columns:
                series = feature_history[feature_name].dropna()
                
                if len(series) >= 30:  # Minimum for meaningful statistical tests
                    # Test for stationarity
                    stationarity_test = self.tester.test_stationarity(series.values)
                    validation_results[feature_name]['stationarity'] = stationarity_test
                    
                    # Test for normality
                    normality_test = self.tester.test_normality(series.values)
                    validation_results[feature_name]['normality'] = normality_test
        
        return validation_results
    
    def generate_features(self, order_book: OrderBook, 
                         price_history: np.ndarray, 
                         volume_history: np.ndarray,
                         validate: bool = True) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        """
        Generate all features for the current market state.
        
        Returns:
            Tuple of (features, validation_results)
        """
        all_features = {}
        
        # Extract different types of features
        if self.config.price_features:
            price_features = self.extract_price_features(order_book, price_history)
            all_features.update(price_features)
        
        if self.config.volume_features:
            volume_features = self.extract_volume_features(order_book, volume_history)
            all_features.update(volume_features)
        
        if self.config.microstructure_features:
            microstructure_features = self.extract_microstructure_features(order_book)
            all_features.update(microstructure_features)
        
        if self.config.technical_features:
            technical_features = self.extract_technical_features(price_history, volume_history)
            all_features.update(technical_features)
        
        # Validate features if requested
        validation_results = None
        if validate:
            validation_results = self.validate_features(all_features)
        
        return all_features, validation_results
    
    def get_feature_importance_stats(self, feature_history: pd.DataFrame, 
                                   target: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using correlation and mutual information."""
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import StandardScaler
        
        importance_stats = {}
        
        for column in feature_history.columns:
            series = feature_history[column].dropna()
            
            if len(series) < 10:  # Not enough data
                continue
                
            # Correlation with target
            if len(series) == len(target):
                correlation = np.corrcoef(series, target)[0, 1]
                importance_stats[f'{column}_correlation'] = correlation
            
            # Mutual information (if we have enough data)
            if len(series) >= 50:
                try:
                    scaler = StandardScaler()
                    scaled_feature = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
                    mi_score = mutual_info_regression(scaled_feature.reshape(-1, 1), target[:len(series)])
                    importance_stats[f'{column}_mutual_info'] = mi_score[0]
                except Exception:
                    pass
        
        return importance_stats