"""
Fear and Greed Index Implementation

This module calculates a Fear and Greed Index for financial markets by analyzing
multiple market indicators and sentiment factors to determine if investors are
driven primarily by fear or greed.

The index uses a 0-100 scale where:
- 0-25: Extreme Fear
- 26-45: Fear
- 46-55: Neutral
- 56-75: Greed
- 76-100: Extreme Greed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import logging
import os
import datetime
from collections import defaultdict
import yfinance as yf
import requests
from typing import Dict, List, Tuple, Union, Optional

from src.sentiment.advanced_emotion_sentiment import EnhancedEmotionalAnalyzer
from src.sentiment.advanced_sentiment import AdvancedSentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fear and Greed classification thresholds
FEAR_GREED_THRESHOLDS = {
    'extreme_fear': (0, 25),
    'fear': (25, 45),
    'neutral': (45, 55),
    'greed': (55, 75),
    'extreme_greed': (75, 100)
}

# Fear and Greed colors for visualization
FEAR_GREED_COLORS = {
    'extreme_fear': '#D62828',    # Dark red
    'fear': '#FF6B6B',            # Light red
    'neutral': '#DBDBDB',         # Gray
    'greed': '#7ED957',           # Light green
    'extreme_greed': '#1E8F4E'    # Dark green
}

class MarketIndicator:
    """Base class for market indicators used in the Fear and Greed Index"""
    
    def __init__(self, name: str, weight: float = 1.0, lookback_period: int = 20):
        """
        Initialize a market indicator
        
        Parameters:
        -----------
        name : str
            The name of the indicator
        weight : float
            The weight of this indicator in the overall index
        lookback_period : int
            The number of trading days to look back for normalization
        """
        self.name = name
        self.weight = weight
        self.lookback_period = lookback_period
        self._value = None
        self._normalized_value = None
        self._last_update = None
    
    def update(self, *args, **kwargs) -> float:
        """
        Update the indicator value
        
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def normalize(self, value: float, history: List[float]) -> float:
        """
        Normalize a value to a 0-100 scale based on historical data
        
        Parameters:
        -----------
        value : float
            The current value to normalize
        history : List[float]
            Historical values for computing the range
            
        Returns:
        --------
        float
            Normalized value (0-100)
        """
        if not history or len(history) < 2:
            logger.warning(f"Not enough historical data to normalize {self.name}")
            return 50.0  # Default to neutral if not enough data
            
        # Remove outliers (using 1.5 * IQR method)
        q1, q3 = np.percentile(history, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_history = [x for x in history if lower_bound <= x <= upper_bound]
        
        if not filtered_history:
            filtered_history = history  # Fall back to original if filtering removed all
            
        min_val = min(filtered_history)
        max_val = max(filtered_history)
        
        # Handle division by zero
        if max_val == min_val:
            return 50.0
            
        # Calculate normalized value (0-100)
        normalized = 100 * (value - min_val) / (max_val - min_val)
        
        # Ensure value is within bounds
        normalized = max(0, min(100, normalized))
        
        return normalized


class MarketVolatilityIndicator(MarketIndicator):
    """Market volatility indicator based on VIX"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 30, inverse: bool = True):
        super().__init__("Market Volatility", weight, lookback_period)
        # Typically, high VIX = fear, so we invert for fear and greed index
        self.inverse = inverse
        
    def update(self, vix_data: pd.DataFrame = None) -> float:
        """
        Update the volatility indicator based on VIX data
        
        Parameters:
        -----------
        vix_data : pd.DataFrame, optional
            VIX data (will fetch if not provided)
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        if vix_data is None:
            try:
                # Fetch VIX data
                vix_data = yf.download('^VIX', period=f"{self.lookback_period + 10}d")
            except Exception as e:
                logger.error(f"Error fetching VIX data: {e}")
                return 50.0
        
        if vix_data.empty:
            logger.warning("Empty VIX data")
            return 50.0
            
        # Get current VIX and historical values
        current_vix = vix_data['Close'].iloc[-1]
        historical_vix = vix_data['Close'].iloc[-self.lookback_period:].tolist()
        
        # Normalize VIX (higher VIX typically means more fear)
        normalized = self.normalize(current_vix, historical_vix)
        
        # Invert if necessary (since higher VIX = more fear)
        if self.inverse:
            normalized = 100 - normalized
            
        self._value = current_vix
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized


class MarketMomentumIndicator(MarketIndicator):
    """Market momentum indicator based on price vs moving average"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 125, moving_avg_period: int = 50):
        super().__init__("Market Momentum", weight, lookback_period)
        self.moving_avg_period = moving_avg_period
        
    def update(self, market_data: pd.DataFrame = None, index_symbol: str = '^GSPC') -> float:
        """
        Update the momentum indicator
        
        Parameters:
        -----------
        market_data : pd.DataFrame, optional
            Market data (will fetch if not provided)
        index_symbol : str
            The market index symbol to use
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        if market_data is None:
            try:
                # Fetch market data (S&P 500 by default)
                market_data = yf.download(index_symbol, period=f"{self.lookback_period + self.moving_avg_period + 10}d")
            except Exception as e:
                logger.error(f"Error fetching market data for {index_symbol}: {e}")
                return 50.0
        
        if market_data.empty:
            logger.warning(f"Empty market data for {index_symbol}")
            return 50.0
            
        # Calculate current price vs moving average
        market_data['MA'] = market_data['Close'].rolling(window=self.moving_avg_period).mean()
        
        # Current price vs moving average as percentage
        current_price = market_data['Close'].iloc[-1]
        current_ma = market_data['MA'].iloc[-1]
        price_vs_ma = ((current_price / current_ma) - 1) * 100
        
        # Get historical values for normalization
        historical_values = []
        for i in range(self.lookback_period):
            if i + self.moving_avg_period < len(market_data):
                price = market_data['Close'].iloc[-(i+1)]
                ma = market_data['MA'].iloc[-(i+1)]
                if not np.isnan(ma) and ma != 0:
                    hist_value = ((price / ma) - 1) * 100
                    historical_values.append(hist_value)
        
        # Normalize the value
        normalized = self.normalize(price_vs_ma, historical_values)
        
        self._value = price_vs_ma
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized


class MarketStrengthIndicator(MarketIndicator):
    """Market strength indicator based on stocks above moving average"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 60, moving_avg_period: int = 50, 
                 stock_universe: List[str] = None, min_stocks: int = 100):
        super().__init__("Market Strength", weight, lookback_period)
        self.moving_avg_period = moving_avg_period
        self.stock_universe = stock_universe or []
        self.min_stocks = min_stocks
        
    def update(self, stock_data: Dict[str, pd.DataFrame] = None) -> float:
        """
        Update the market strength indicator
        
        Parameters:
        -----------
        stock_data : Dict[str, pd.DataFrame], optional
            Stock data for multiple tickers (will fetch if not provided)
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        # If no stock data and no universe defined, use default list
        if stock_data is None and not self.stock_universe:
            # Use stocks from S&P 500 as default universe (simplified for example)
            self.stock_universe = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", 
                                  "JPM", "BAC", "WMT", "JNJ", "PG", "V", "MA", "UNH", 
                                  "NVDA", "HD", "PFE", "INTC", "VZ", "DIS"]
        
        # Fetch stock data if not provided
        if stock_data is None:
            stock_data = {}
            valid_tickers = []
            
            for ticker in self.stock_universe:
                try:
                    data = yf.download(ticker, period=f"{self.lookback_period + self.moving_avg_period + 10}d", progress=False)
                    if not data.empty:
                        stock_data[ticker] = data
                        valid_tickers.append(ticker)
                except Exception as e:
                    logger.debug(f"Error fetching data for {ticker}: {e}")
            
            self.stock_universe = valid_tickers
        
        if not stock_data or len(stock_data) < self.min_stocks:
            logger.warning(f"Not enough stock data for market strength analysis: {len(stock_data)} < {self.min_stocks}")
            return 50.0
        
        # Calculate percentage of stocks above their moving average
        above_ma_pct = self._calculate_stocks_above_ma(stock_data)
        
        # Get historical values for normalization
        historical_values = []
        for i in range(1, min(self.lookback_period, 15)):  # Limit lookback to avoid excessive computation
            hist_pct = self._calculate_stocks_above_ma(stock_data, offset=i)
            if hist_pct is not None:
                historical_values.append(hist_pct)
        
        # Add some synthetic values if we don't have enough history
        if len(historical_values) < 5:
            historical_values.extend([30, 50, 70])
        
        # Normalize the value
        normalized = self.normalize(above_ma_pct, historical_values)
        
        self._value = above_ma_pct
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized
    
    def _calculate_stocks_above_ma(self, stock_data: Dict[str, pd.DataFrame], offset: int = 0) -> float:
        """
        Calculate percentage of stocks above their moving average
        
        Parameters:
        -----------
        stock_data : Dict[str, pd.DataFrame]
            Stock data for multiple tickers
        offset : int
            Offset from the most recent data point (for historical calculations)
            
        Returns:
        --------
        float
            Percentage of stocks above their moving average
        """
        stocks_above_ma = 0
        total_valid_stocks = 0
        
        for ticker, data in stock_data.items():
            if len(data) <= self.moving_avg_period + offset:
                continue
                
            # Calculate moving average
            data['MA'] = data['Close'].rolling(window=self.moving_avg_period).mean()
            
            try:
                # Compare current price to moving average
                current_price = data['Close'].iloc[-(offset+1)]
                current_ma = data['MA'].iloc[-(offset+1)]
                
                if not np.isnan(current_ma) and current_price > current_ma:
                    stocks_above_ma += 1
                
                total_valid_stocks += 1
            except (IndexError, KeyError):
                continue
        
        if total_valid_stocks == 0:
            return None
            
        # Return percentage
        return (stocks_above_ma / total_valid_stocks) * 100


class MarketBreadthIndicator(MarketIndicator):
    """Market breadth indicator based on advancing vs declining issues"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 30):
        super().__init__("Market Breadth", weight, lookback_period)
        
    def update(self, advance_decline_data: pd.DataFrame = None) -> float:
        """
        Update the market breadth indicator
        
        Parameters:
        -----------
        advance_decline_data : pd.DataFrame, optional
            Advance/decline data (must fetch externally)
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        # This typically requires subscription to a data provider
        # Demonstrating with a simplified version
        if advance_decline_data is None:
            try:
                # Try to use NYSE advance-decline data if available
                # This is a placeholder - real implementation would use a proper data source
                nyse_advance_decline = None
                
                # If data not available, fall back to simulating with S&P 500
                sp500_data = yf.download('^GSPC', period=f"{self.lookback_period + 10}d")
                if sp500_data.empty:
                    logger.warning("Empty S&P 500 data for market breadth estimation")
                    return 50.0
                
                # Create synthetic advance/decline data based on daily price changes
                advance_decline_data = pd.DataFrame()
                advance_decline_data['Date'] = sp500_data.index
                advance_decline_data['Advances'] = 0
                advance_decline_data['Declines'] = 0
                
                # Simulate advances and declines based on S&P 500 performance
                for i in range(1, len(sp500_data)):
                    if sp500_data['Close'].iloc[i] > sp500_data['Close'].iloc[i-1]:
                        # On up days, assume 60-80% of stocks advance
                        advance_pct = np.random.uniform(0.6, 0.8)
                    else:
                        # On down days, assume 30-45% of stocks advance
                        advance_pct = np.random.uniform(0.3, 0.45)
                    
                    # Assume 500 stocks total (like S&P 500)
                    total_stocks = 500
                    advance_decline_data.at[advance_decline_data.index[i], 'Advances'] = int(advance_pct * total_stocks)
                    advance_decline_data.at[advance_decline_data.index[i], 'Declines'] = total_stocks - int(advance_pct * total_stocks)
                
            except Exception as e:
                logger.error(f"Error creating synthetic market breadth data: {e}")
                return 50.0
        
        # Calculate advance-decline ratio
        advance_decline_data['AD_Ratio'] = advance_decline_data['Advances'] / advance_decline_data['Declines']
        
        # Get current and historical values
        current_ratio = advance_decline_data['AD_Ratio'].iloc[-1]
        historical_ratios = advance_decline_data['AD_Ratio'].iloc[-self.lookback_period:].tolist()
        
        # Normalize the ratio
        normalized = self.normalize(current_ratio, historical_ratios)
        
        self._value = current_ratio
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized


class SafeHavenDemandIndicator(MarketIndicator):
    """Safe haven demand indicator based on Treasury yields vs S&P 500"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 30):
        super().__init__("Safe Haven Demand", weight, lookback_period)
        
    def update(self, sp500_data: pd.DataFrame = None, treasury_data: pd.DataFrame = None) -> float:
        """
        Update the safe haven demand indicator
        
        Parameters:
        -----------
        sp500_data : pd.DataFrame, optional
            S&P 500 data (will fetch if not provided)
        treasury_data : pd.DataFrame, optional
            10-year Treasury yield data (will fetch if not provided)
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        # Fetch data if not provided
        if sp500_data is None:
            try:
                sp500_data = yf.download('^GSPC', period=f"{self.lookback_period + 10}d")
            except Exception as e:
                logger.error(f"Error fetching S&P 500 data: {e}")
                return 50.0
        
        if treasury_data is None:
            try:
                # 10-year Treasury Yield
                treasury_data = yf.download('^TNX', period=f"{self.lookback_period + 10}d")
            except Exception as e:
                logger.error(f"Error fetching Treasury data: {e}")
                return 50.0
        
        if sp500_data.empty or treasury_data.empty:
            logger.warning("Empty data for safe haven demand analysis")
            return 50.0
        
        # Calculate ratio of S&P 500 to Treasury yields (inverse relationship - high yields should correspond to low stock prices in fear periods)
        # Align dates
        common_dates = sp500_data.index.intersection(treasury_data.index)
        if len(common_dates) < 2:
            logger.warning("Not enough common dates for S&P 500 and Treasury data")
            return 50.0
            
        sp500_aligned = sp500_data.loc[common_dates]
        treasury_aligned = treasury_data.loc[common_dates]
        
        # Calculate ratio (higher values typically indicate more greed/risk-on)
        ratio_data = pd.DataFrame(index=common_dates)
        ratio_data['SP500_Norm'] = sp500_aligned['Close'] / sp500_aligned['Close'].iloc[0]
        ratio_data['Treasury_Norm'] = treasury_aligned['Close'] / treasury_aligned['Close'].iloc[0]
        ratio_data['Ratio'] = ratio_data['SP500_Norm'] / ratio_data['Treasury_Norm']
        
        # Get current ratio and historical values
        current_ratio = ratio_data['Ratio'].iloc[-1]
        historical_ratios = ratio_data['Ratio'].iloc[-self.lookback_period:].tolist()
        
        # Normalize the ratio
        normalized = self.normalize(current_ratio, historical_ratios)
        
        self._value = current_ratio
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized


class JunkBondDemandIndicator(MarketIndicator):
    """Junk bond demand indicator based on high-yield vs investment-grade spreads"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 30):
        super().__init__("Junk Bond Demand", weight, lookback_period)
        
    def update(self, high_yield_data: pd.DataFrame = None, investment_grade_data: pd.DataFrame = None) -> float:
        """
        Update the junk bond demand indicator
        
        Parameters:
        -----------
        high_yield_data : pd.DataFrame, optional
            High-yield bond ETF data (will fetch if not provided)
        investment_grade_data : pd.DataFrame, optional
            Investment-grade bond ETF data (will fetch if not provided)
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        # Fetch data if not provided (using ETFs as proxies)
        if high_yield_data is None:
            try:
                # HYG for high yield corporate bonds
                high_yield_data = yf.download('HYG', period=f"{self.lookback_period + 10}d")
            except Exception as e:
                logger.error(f"Error fetching high-yield bond data: {e}")
                return 50.0
        
        if investment_grade_data is None:
            try:
                # LQD for investment grade corporate bonds
                investment_grade_data = yf.download('LQD', period=f"{self.lookback_period + 10}d")
            except Exception as e:
                logger.error(f"Error fetching investment-grade bond data: {e}")
                return 50.0
        
        if high_yield_data.empty or investment_grade_data.empty:
            logger.warning("Empty data for junk bond demand analysis")
            return 50.0
        
        # Calculate ratio of high-yield to investment-grade performance
        # Align dates
        common_dates = high_yield_data.index.intersection(investment_grade_data.index)
        if len(common_dates) < 2:
            logger.warning("Not enough common dates for bond data")
            return 50.0
            
        high_yield_aligned = high_yield_data.loc[common_dates]
        investment_grade_aligned = investment_grade_data.loc[common_dates]
        
        # Calculate ratio (higher values typically indicate more greed/risk-on)
        ratio_data = pd.DataFrame(index=common_dates)
        ratio_data['HY_Norm'] = high_yield_aligned['Close'] / high_yield_aligned['Close'].iloc[0]
        ratio_data['IG_Norm'] = investment_grade_aligned['Close'] / investment_grade_aligned['Close'].iloc[0]
        ratio_data['Ratio'] = ratio_data['HY_Norm'] / ratio_data['IG_Norm']
        
        # Get current ratio and historical values
        current_ratio = ratio_data['Ratio'].iloc[-1]
        historical_ratios = ratio_data['Ratio'].iloc[-self.lookback_period:].tolist()
        
        # Normalize the ratio
        normalized = self.normalize(current_ratio, historical_ratios)
        
        self._value = current_ratio
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized


class SentimentIndicator(MarketIndicator):
    """Market sentiment indicator based on text sentiment analysis"""
    
    def __init__(self, weight: float = 1.0, lookback_period: int = 30, sentiment_analyzer=None):
        super().__init__("News Sentiment", weight, lookback_period)
        self.sentiment_analyzer = sentiment_analyzer or AdvancedSentimentAnalyzer()
        
    def update(self, news_data: List[str] = None, social_data: List[str] = None) -> float:
        """
        Update the sentiment indicator based on news and social media
        
        Parameters:
        -----------
        news_data : List[str], optional
            List of news headlines or articles
        social_data : List[str], optional
            List of social media posts
            
        Returns:
        --------
        float
            The normalized indicator value (0-100)
        """
        # If no data provided, try to use available data from files
        if news_data is None and social_data is None:
            news_data = self._load_recent_news()
            social_data = self._load_recent_social()
            
        if not news_data and not social_data:
            logger.warning("No text data available for sentiment analysis")
            return 50.0
            
        # Combine all text data
        all_texts = (news_data or []) + (social_data or [])
        
        # Calculate sentiment scores
        sentiment_scores = []
        for text in all_texts:
            if not text or len(text) < 10:
                continue
                
            try:
                sentiment_result = self.sentiment_analyzer.analyze_text(text)
                if isinstance(sentiment_result, dict) and 'score' in sentiment_result:
                    sentiment_scores.append(sentiment_result['score'])
                elif hasattr(sentiment_result, 'score'):
                    sentiment_scores.append(sentiment_result.score)
            except Exception as e:
                logger.debug(f"Error analyzing sentiment: {e}")
                
        if not sentiment_scores:
            logger.warning("No valid sentiment scores generated")
            return 50.0
            
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Convert from -1 to 1 scale to 0 to 100 scale
        normalized = (avg_sentiment + 1) * 50
        
        self._value = avg_sentiment
        self._normalized_value = normalized
        self._last_update = datetime.datetime.now()
        
        return normalized
    
    def _load_recent_news(self) -> List[str]:
        """Load recent news from available files"""
        news_texts = []
        
        try:
            news_dir = os.path.join('data', 'news')
            if os.path.exists(news_dir):
                # Get the latest news files
                news_files = [f for f in os.listdir(news_dir) if f.endswith('.csv')]
                news_files.sort(reverse=True)  # Latest first
                
                # Load up to 5 latest news files
                for file in news_files[:5]:
                    file_path = os.path.join(news_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        if 'title' in df.columns and 'description' in df.columns:
                            for _, row in df.iterrows():
                                news_texts.append(f"{row['title']}. {row['description']}")
                    except Exception as e:
                        logger.debug(f"Error loading news file {file}: {e}")
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            
        return news_texts
    
    def _load_recent_social(self) -> List[str]:
        """Load recent social media posts from available files"""
        social_texts = []
        
        try:
            reddit_dir = os.path.join('data', 'reddit')
            if os.path.exists(reddit_dir):
                # Get the latest reddit files
                reddit_files = [f for f in os.listdir(reddit_dir) if f.endswith('.csv')]
                reddit_files.sort(reverse=True)  # Latest first
                
                # Load up to 3 latest reddit files
                for file in reddit_files[:3]:
                    file_path = os.path.join(reddit_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        if 'text' in df.columns:
                            social_texts.extend(df['text'].dropna().tolist())
                        elif 'title' in df.columns:
                            social_texts.extend(df['title'].dropna().tolist())
                    except Exception as e:
                        logger.debug(f"Error loading reddit file {file}: {e}")
        except Exception as e:
            logger.error(f"Error loading social media data: {e}")
            
        return social_texts


class FearGreedIndex:
    """
    Fear and Greed Index implementation that combines multiple market indicators
    to create a comprehensive gauge of market sentiment
    """
    
    def __init__(self, include_sentiment: bool = True, auto_update: bool = True):
        """
        Initialize the Fear and Greed Index
        
        Parameters:
        -----------
        include_sentiment : bool
            Whether to include text sentiment analysis
        auto_update : bool
            Whether to update the index automatically at initialization
        """
        # Initialize indicators with weights
        self.indicators = {
            'volatility': MarketVolatilityIndicator(weight=1.0),
            'momentum': MarketMomentumIndicator(weight=1.0),
            'strength': MarketStrengthIndicator(weight=0.8),
            'breadth': MarketBreadthIndicator(weight=0.8),
            'safe_haven': SafeHavenDemandIndicator(weight=0.7),
            'junk_bond': JunkBondDemandIndicator(weight=0.7)
        }
        
        # Add sentiment indicator if requested
        if include_sentiment:
            try:
                # Try to use enhanced emotional analyzer if available
                self.indicators['sentiment'] = SentimentIndicator(
                    weight=0.5,
                    sentiment_analyzer=EnhancedEmotionalAnalyzer()
                )
            except Exception:
                # Fall back to basic sentiment analyzer
                self.indicators['sentiment'] = SentimentIndicator(weight=0.5)
        
        # Initialize index data
        self.current_index = None
        self.current_classification = None
        self.history = []
        self.last_update = None
        
        # Auto-update if requested
        if auto_update:
            self.update()
            
    def update(self) -> Dict:
        """
        Update the Fear and Greed Index by recalculating all indicators
        
        Returns:
        --------
        Dict
            Dictionary with the current index value and classification
        """
        indicator_values = {}
        total_weight = 0
        
        # Update each indicator
        for name, indicator in self.indicators.items():
            try:
                normalized_value = indicator.update()
                indicator_values[name] = {
                    'value': indicator._value,
                    'normalized': normalized_value,
                    'weight': indicator.weight
                }
                total_weight += indicator.weight
            except Exception as e:
                logger.error(f"Error updating {name} indicator: {e}")
        
        # Calculate weighted average
        if total_weight > 0:
            weighted_sum = sum(v['normalized'] * v['weight'] for v in indicator_values.values())
            index_value = weighted_sum / total_weight
        else:
            logger.warning("No valid indicators for Fear and Greed Index")
            index_value = 50.0  # Default to neutral
            
        # Determine classification
        classification = self._classify_index(index_value)
        
        # Store current values
        self.current_index = index_value
        self.current_classification = classification
        self.last_update = datetime.datetime.now()
        
        # Add to history (limited to last 100 values)
        self.history.append({
            'date': self.last_update,
            'index': index_value,
            'classification': classification,
            'indicators': indicator_values
        })
        if len(self.history) > 100:
            self.history = self.history[-100:]
            
        return {
            'index': index_value,
            'classification': classification,
            'indicators': indicator_values,
            'updated_at': self.last_update
        }
    
    def _classify_index(self, index_value: float) -> str:
        """
        Classify the index value into a fear/greed category
        
        Parameters:
        -----------
        index_value : float
            The index value to classify
            
        Returns:
        --------
        str
            Classification (extreme_fear, fear, neutral, greed, extreme_greed)
        """
        for classification, (lower, upper) in FEAR_GREED_THRESHOLDS.items():
            if lower <= index_value < upper:
                return classification
                
        # Default to neutral if something goes wrong
        return 'neutral'
    
    def get_current_index(self) -> Dict:
        """
        Get the current Fear and Greed Index value and classification
        
        Returns:
        --------
        Dict
            Dictionary with the current index value and classification
        """
        if self.current_index is None:
            self.update()
            
        return {
            'index': self.current_index,
            'classification': self.current_classification,
            'updated_at': self.last_update
        }
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical Fear and Greed Index data
        
        Parameters:
        -----------
        days : int
            Number of days of historical data to return
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with historical index values
        """
        if not self.history:
            return pd.DataFrame()
            
        # Convert history to DataFrame
        df = pd.DataFrame([
            {'date': h['date'], 'index': h['index'], 'classification': h['classification']}
            for h in self.history
        ])
        
        # Limit to requested number of days
        if len(df) > days:
            df = df.tail(days)
            
        return df
    
    def plot_gauge(self, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> None:
        """
        Plot the Fear and Greed Index as a gauge
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.current_index is None:
            self.update()
            
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        
        # Convert index to angle (0-100 to -π/2 to π/2)
        theta = np.radians(90 - self.current_index * 1.8)
        
        # Set up gauge properties
        gauge_min = np.radians(90)  # -π/2
        gauge_max = np.radians(-90)  # π/2
        
        # Create gauge background
        color_stops = [
            (0.0, FEAR_GREED_COLORS['extreme_fear']),
            (0.25, FEAR_GREED_COLORS['fear']),
            (0.45, FEAR_GREED_COLORS['neutral']),
            (0.55, FEAR_GREED_COLORS['neutral']),
            (0.75, FEAR_GREED_COLORS['greed']),
            (1.0, FEAR_GREED_COLORS['extreme_greed'])
        ]
        
        # Create custom colormap
        cmap = mcolors.LinearSegmentedColormap.from_list("fear_greed", color_stops)
        
        # Draw gauge background
        angles = np.linspace(gauge_min, gauge_max, 100)
        norms = np.linspace(0, 1, len(angles))
        
        # Draw colored segments for each classification
        for classification, (lower, upper) in FEAR_GREED_THRESHOLDS.items():
            # Convert to angles
            angle_lower = np.radians(90 - lower * 1.8)
            angle_upper = np.radians(90 - upper * 1.8)
            
            # Create patch for this segment
            wedge = Wedge((0, 0), 0.8, np.degrees(angle_lower), np.degrees(angle_upper), 
                         width=0.2, color=FEAR_GREED_COLORS[classification])
            ax.add_patch(wedge)
        
        # Draw needle
        ax.plot([0, np.cos(theta)], [0, np.sin(theta)], 'k-', lw=3)
        ax.plot([0, 0], [0, 0], 'ko', markersize=10)
        
        # Add text in center
        ax.text(0, -0.15, f"{self.current_index:.1f}", ha='center', va='center',
               fontsize=24, fontweight='bold')
        ax.text(0, -0.25, f"{self.current_classification.replace('_', ' ').title()}", 
               ha='center', va='center', fontsize=16)
        
        # Set up plot styling
        ax.set_title("Fear & Greed Index", fontsize=20, pad=20)
        ax.set_axis_off()
        
        # Add labels at key points
        ax.text(0.85*np.cos(np.radians(90)), 0.85*np.sin(np.radians(90)), 
               "Extreme\nFear", ha='center', va='center', fontsize=12)
        ax.text(0.85*np.cos(np.radians(45)), 0.85*np.sin(np.radians(45)), 
               "Fear", ha='center', va='center', fontsize=12)
        ax.text(0.85*np.cos(np.radians(0)), 0.85*np.sin(np.radians(0)), 
               "Neutral", ha='center', va='center', fontsize=12)
        ax.text(0.85*np.cos(np.radians(-45)), 0.85*np.sin(np.radians(-45)), 
               "Greed", ha='center', va='center', fontsize=12)
        ax.text(0.85*np.cos(np.radians(-90)), 0.85*np.sin(np.radians(-90)), 
               "Extreme\nGreed", ha='center', va='center', fontsize=12)
        
        # Add date of last update
        last_update_str = self.last_update.strftime("%Y-%m-%d %H:%M")
        ax.text(0, -0.45, f"Last Updated: {last_update_str}", ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_history(self, days: int = 30, figsize: Tuple[int, int] = (12, 6), 
                    save_path: Optional[str] = None) -> None:
        """
        Plot historical Fear and Greed Index data
        
        Parameters:
        -----------
        days : int
            Number of days of history to plot
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        df = self.get_historical_data(days)
        
        if df.empty:
            logger.warning("No historical data to plot")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color map for line
        cmap = plt.cm.RdYlGn  # Red to yellow to green
        norm = plt.Normalize(0, 100)
        
        # Plot the line with color gradient
        for i in range(len(df) - 1):
            x = df['date'].iloc[i:i+2]
            y = df['index'].iloc[i:i+2]
            color = cmap(norm(y.iloc[0]))
            ax.plot(x, y, color=color, linewidth=3)
            
        # Add points
        scatter = ax.scatter(df['date'], df['index'], c=df['index'], 
                           cmap=cmap, norm=norm, s=50, zorder=5)
        
        # Add background colors for each classification
        ax.axhspan(FEAR_GREED_THRESHOLDS['extreme_fear'][0], 
                  FEAR_GREED_THRESHOLDS['extreme_fear'][1], 
                  facecolor=FEAR_GREED_COLORS['extreme_fear'], alpha=0.2)
        ax.axhspan(FEAR_GREED_THRESHOLDS['fear'][0], 
                  FEAR_GREED_THRESHOLDS['fear'][1], 
                  facecolor=FEAR_GREED_COLORS['fear'], alpha=0.2)
        ax.axhspan(FEAR_GREED_THRESHOLDS['neutral'][0], 
                  FEAR_GREED_THRESHOLDS['neutral'][1], 
                  facecolor=FEAR_GREED_COLORS['neutral'], alpha=0.2)
        ax.axhspan(FEAR_GREED_THRESHOLDS['greed'][0], 
                  FEAR_GREED_THRESHOLDS['greed'][1], 
                  facecolor=FEAR_GREED_COLORS['greed'], alpha=0.2)
        ax.axhspan(FEAR_GREED_THRESHOLDS['extreme_greed'][0], 
                  FEAR_GREED_THRESHOLDS['extreme_greed'][1], 
                  facecolor=FEAR_GREED_COLORS['extreme_greed'], alpha=0.2)
        
        # Add labels for regions
        ax.text(df['date'].iloc[0], 12.5, "Extreme Fear", fontsize=10, 
               ha='left', va='center')
        ax.text(df['date'].iloc[0], 35, "Fear", fontsize=10, 
               ha='left', va='center')
        ax.text(df['date'].iloc[0], 50, "Neutral", fontsize=10, 
               ha='left', va='center')
        ax.text(df['date'].iloc[0], 65, "Greed", fontsize=10, 
               ha='left', va='center')
        ax.text(df['date'].iloc[0], 87.5, "Extreme Greed", fontsize=10, 
               ha='left', va='center')
        
        # Set up plot styling
        ax.set_title("Historical Fear & Greed Index", fontsize=16)
        ax.set_ylabel("Index Value", fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Fear & Greed Value")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_indicators(self, figsize: Tuple[int, int] = (10, 8), 
                       save_path: Optional[str] = None) -> None:
        """
        Plot each indicator's contribution to the Fear and Greed Index
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.history or not self.history[-1]['indicators']:
            logger.warning("No indicator data to plot")
            return
            
        # Get latest indicator values
        indicators = self.history[-1]['indicators']
        
        # Create data for plotting
        names = []
        values = []
        colors = []
        
        for name, data in indicators.items():
            names.append(name.replace('_', ' ').title())
            values.append(data['normalized'])
            
            # Determine color based on value
            for classification, (lower, upper) in FEAR_GREED_THRESHOLDS.items():
                if lower <= data['normalized'] < upper:
                    colors.append(FEAR_GREED_COLORS[classification])
                    break
            else:
                colors.append(FEAR_GREED_COLORS['neutral'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        bars = ax.barh(names, values, color=colors)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f"{width:.1f}", ha='left', va='center')
        
        # Add background colors for each classification
        ax.axvspan(FEAR_GREED_THRESHOLDS['extreme_fear'][0], 
                  FEAR_GREED_THRESHOLDS['extreme_fear'][1], 
                  facecolor=FEAR_GREED_COLORS['extreme_fear'], alpha=0.1)
        ax.axvspan(FEAR_GREED_THRESHOLDS['fear'][0], 
                  FEAR_GREED_THRESHOLDS['fear'][1], 
                  facecolor=FEAR_GREED_COLORS['fear'], alpha=0.1)
        ax.axvspan(FEAR_GREED_THRESHOLDS['neutral'][0], 
                  FEAR_GREED_THRESHOLDS['neutral'][1], 
                  facecolor=FEAR_GREED_COLORS['neutral'], alpha=0.1)
        ax.axvspan(FEAR_GREED_THRESHOLDS['greed'][0], 
                  FEAR_GREED_THRESHOLDS['greed'][1], 
                  facecolor=FEAR_GREED_COLORS['greed'], alpha=0.1)
        ax.axvspan(FEAR_GREED_THRESHOLDS['extreme_greed'][0], 
                  FEAR_GREED_THRESHOLDS['extreme_greed'][1], 
                  facecolor=FEAR_GREED_COLORS['extreme_greed'], alpha=0.1)
        
        # Set up plot styling
        ax.set_title("Fear & Greed Index Indicators", fontsize=16)
        ax.set_xlabel("Indicator Value (0-100)", fontsize=12)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add labels for regions at the top
        ax.text(12.5, len(names) + 0.3, "Extreme\nFear", fontsize=9, 
               ha='center', va='bottom')
        ax.text(35, len(names) + 0.3, "Fear", fontsize=9, 
               ha='center', va='bottom')
        ax.text(50, len(names) + 0.3, "Neutral", fontsize=9, 
               ha='center', va='bottom')
        ax.text(65, len(names) + 0.3, "Greed", fontsize=9, 
               ha='center', va='bottom')
        ax.text(87.5, len(names) + 0.3, "Extreme\nGreed", fontsize=9, 
               ha='center', va='bottom')
        
        # Add date of last update
        last_update_str = self.last_update.strftime("%Y-%m-%d %H:%M")
        ax.text(0, -0.8, f"Last Updated: {last_update_str}", transform=ax.transAxes, 
               fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()


def calculate_fear_greed_index() -> Dict:
    """
    Calculate the current Fear and Greed Index
    
    Returns:
    --------
    Dict
        Dictionary with the current index value and classification
    """
    # Initialize and update the Fear and Greed Index
    fg_index = FearGreedIndex(include_sentiment=True)
    result = fg_index.update()
    
    # Generate and save plots
    try:
        output_dir = os.path.join('data', 'analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save gauge plot
        gauge_path = os.path.join(output_dir, 'fear_greed_gauge.png')
        fg_index.plot_gauge(save_path=gauge_path)
        
        # Save history plot if there's enough data
        if len(fg_index.history) > 2:
            history_path = os.path.join(output_dir, 'fear_greed_history.png')
            fg_index.plot_history(save_path=history_path)
        
        # Save indicators plot
        indicators_path = os.path.join(output_dir, 'fear_greed_indicators.png')
        fg_index.plot_indicators(save_path=indicators_path)
    except Exception as e:
        logger.error(f"Error generating Fear and Greed plots: {e}")
    
    return result


# Example usage
if __name__ == "__main__":
    # Calculate the current Fear and Greed Index
    result = calculate_fear_greed_index()
    
    # Print the results
    print("\n" + "="*50)
    print(f"Fear and Greed Index: {result['index']:.1f}")
    print(f"Classification: {result['classification'].replace('_', ' ').title()}")
    print("="*50)
    
    # Print indicator details
    print("\nIndicator Details:")
    for name, data in result['indicators'].items():
        if 'normalized' in data:
            print(f"  {name.replace('_', ' ').title()}: {data['normalized']:.1f}")
    
    print("\nImages saved to data/analysis/ directory")