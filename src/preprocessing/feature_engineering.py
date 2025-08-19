"""
Feature engineering for stock sentiment analysis
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import talib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class StockFeatureEngineer:
    """
    Class for generating features for stock sentiment analysis
    """
    def __init__(self):
        """
        Initialize feature engineer
        """
        pass
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the price data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added technical indicators
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid DataFrame")
            return df
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        try:
            # Convert price data to numpy float64 arrays (double precision) for TA-Lib
            open_values = result['Open'].values.astype(np.float64)
            high_values = result['High'].values.astype(np.float64)
            low_values = result['Low'].values.astype(np.float64)
            close_values = result['Close'].values.astype(np.float64)
            volume_values = result['Volume'].values.astype(np.float64)
            
            # Moving Averages
            result['SMA_5'] = talib.SMA(close_values, timeperiod=5)
            result['SMA_10'] = talib.SMA(close_values, timeperiod=10)
            result['SMA_20'] = talib.SMA(close_values, timeperiod=20)
            result['SMA_50'] = talib.SMA(close_values, timeperiod=50)
            result['SMA_200'] = talib.SMA(close_values, timeperiod=200)
            
            # Exponential Moving Averages
            result['EMA_5'] = talib.EMA(close_values, timeperiod=5)
            result['EMA_10'] = talib.EMA(close_values, timeperiod=10)
            result['EMA_20'] = talib.EMA(close_values, timeperiod=20)
            result['EMA_50'] = talib.EMA(close_values, timeperiod=50)
            result['EMA_200'] = talib.EMA(close_values, timeperiod=200)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close_values, fastperiod=12, slowperiod=26, signalperiod=9
            )
            result['MACD'] = macd
            result['MACD_Signal'] = macd_signal
            result['MACD_Hist'] = macd_hist
            
            # RSI
            result['RSI_14'] = talib.RSI(close_values, timeperiod=14)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                close_values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            result['BB_Upper'] = upper
            result['BB_Middle'] = middle
            result['BB_Lower'] = lower
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                high_values, low_values, close_values,
                fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            )
            result['Stoch_K'] = slowk
            result['Stoch_D'] = slowd
            
            # Average Directional Index
            result['ADX'] = talib.ADX(
                high_values, low_values, close_values, timeperiod=14
            )
            
            # Commodity Channel Index
            result['CCI'] = talib.CCI(
                high_values, low_values, close_values, timeperiod=14
            )
            
            # On-Balance Volume
            result['OBV'] = talib.OBV(close_values, volume_values)
            
            # Average True Range
            result['ATR'] = talib.ATR(
                high_values, low_values, close_values, timeperiod=14
            )
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} technical indicators")
            return result
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            # Return original DataFrame if error
            return df
    
    def add_price_features(self, df):
        """
        Add price-based features such as returns and volatility
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added price features
        """
        if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
            logger.error("Invalid DataFrame or missing 'Close' column")
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        try:
            # Daily returns
            result['Return_1d'] = result['Close'].pct_change(periods=1)
            
            # Multiple day returns
            result['Return_3d'] = result['Close'].pct_change(periods=3)
            result['Return_5d'] = result['Close'].pct_change(periods=5)
            result['Return_10d'] = result['Close'].pct_change(periods=10)
            result['Return_20d'] = result['Close'].pct_change(periods=20)
            
            # Rolling volatility
            result['Volatility_5d'] = result['Return_1d'].rolling(window=5).std()
            result['Volatility_10d'] = result['Return_1d'].rolling(window=10).std()
            result['Volatility_20d'] = result['Return_1d'].rolling(window=20).std()
            
            # Log returns
            result['Log_Return_1d'] = np.log(result['Close'] / result['Close'].shift(1))
            
            # Price momentum
            result['Momentum_5d'] = result['Close'] / result['Close'].shift(5) - 1
            result['Momentum_10d'] = result['Close'] / result['Close'].shift(10) - 1
            result['Momentum_20d'] = result['Close'] / result['Close'].shift(20) - 1
            
            # Gap up/down
            result['Gap'] = result['Open'] / result['Close'].shift(1) - 1
            
            # High-Low range
            result['HL_Range'] = (result['High'] - result['Low']) / result['Close'].shift(1)
            
            # Relative strength (ratio of stock's return to market return)
            # This would require market index data, placeholder for now
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} price features")
            return result
            
        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            # Return original DataFrame if error
            return df
    
    def add_sentiment_features(self, df):
        """
        Add sentiment-derived features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with sentiment data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sentiment features
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid DataFrame")
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        try:
            # Find sentiment columns
            sentiment_columns = [col for col in result.columns if 'sentiment' in col.lower() and '_compound' in col.lower()]
            
            if not sentiment_columns:
                logger.warning("No sentiment columns found")
                return result
            
            # For each sentiment column, add rolling features
            for col in sentiment_columns:
                base_name = col.replace('_compound', '')
                
                # Rolling mean sentiment
                result[f"{base_name}_3d_mean"] = result[col].rolling(window=3).mean()
                result[f"{base_name}_5d_mean"] = result[col].rolling(window=5).mean()
                
                # Rolling volatility of sentiment
                result[f"{base_name}_3d_std"] = result[col].rolling(window=3).std()
                result[f"{base_name}_5d_std"] = result[col].rolling(window=5).std()
                
                # Sentiment change
                result[f"{base_name}_change_1d"] = result[col].diff(1)
                
                # Sentiment acceleration
                result[f"{base_name}_accel"] = result[f"{base_name}_change_1d"].diff(1)
            
            # If we have both reddit and news sentiment, add combined features
            reddit_col = next((col for col in sentiment_columns if 'reddit' in col.lower()), None)
            news_col = next((col for col in sentiment_columns if 'news' in col.lower()), None)
            
            if reddit_col and news_col:
                # Combined sentiment (weighted average)
                result['combined_sentiment'] = (result[reddit_col] + result[news_col]) / 2
                
                # Sentiment divergence (absolute difference)
                result['sentiment_divergence'] = (result[reddit_col] - result[news_col]).abs()
                
                # Sentiment agreement (product)
                result['sentiment_agreement'] = result[reddit_col] * result[news_col]
                
                # Add rolling features for combined sentiment
                result['combined_sentiment_3d_mean'] = result['combined_sentiment'].rolling(window=3).mean()
                result['combined_sentiment_5d_mean'] = result['combined_sentiment'].rolling(window=5).mean()
                result['combined_sentiment_change_1d'] = result['combined_sentiment'].diff(1)
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} sentiment features")
            return result
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            # Return original DataFrame if error
            return df
    
    def add_target_variables(self, df):
        """
        Add target variables for prediction
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added target variables
        """
        if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
            logger.error("Invalid DataFrame or missing 'Close' column")
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        try:
            # Future returns for different horizons
            result['Target_Return_1d'] = result['Close'].pct_change(periods=1).shift(-1)
            result['Target_Return_3d'] = result['Close'].pct_change(periods=3).shift(-3)
            result['Target_Return_5d'] = result['Close'].pct_change(periods=5).shift(-5)
            
            # Binary direction targets
            result['Target_Direction_1d'] = (result['Target_Return_1d'] > 0).astype(int)
            result['Target_Direction_3d'] = (result['Target_Return_3d'] > 0).astype(int)
            result['Target_Direction_5d'] = (result['Target_Return_5d'] > 0).astype(int)
            
            # Multi-class targets (strong up, up, flat, down, strong down)
            def classify_return(x):
                if x > 0.02:  # Strong up
                    return 2
                elif x > 0.001:  # Up
                    return 1
                elif x < -0.02:  # Strong down
                    return -2
                elif x < -0.001:  # Down
                    return -1
                else:  # Flat
                    return 0
            
            result['Target_Class_1d'] = result['Target_Return_1d'].apply(classify_return)
            result['Target_Class_3d'] = result['Target_Return_3d'].apply(classify_return)
            result['Target_Class_5d'] = result['Target_Return_5d'].apply(classify_return)
            
            logger.info(f"Added {len(result.columns) - len(df.columns)} target variables")
            return result
            
        except Exception as e:
            logger.error(f"Error adding target variables: {e}")
            # Return original DataFrame if error
            return df
    
    def run_complete_feature_engineering(self, df):
        """
        Run complete feature engineering pipeline
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with raw data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all engineered features
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid DataFrame")
            return df
        
        logger.info(f"Running complete feature engineering on DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # Step 1: Add technical indicators
        result = self.add_technical_indicators(df)
        
        # Step 2: Add price features
        result = self.add_price_features(result)
        
        # Step 3: Add sentiment features
        result = self.add_sentiment_features(result)
        
        # Step 4: Add target variables
        result = self.add_target_variables(result)
        
        # Calculate how many features were added
        added_features = len(result.columns) - len(df.columns)
        logger.info(f"Feature engineering complete. Added {added_features} features")
        
        return result
    
    def add_all_features(self, df, include_sentiment=True):
        """
        Add all features to the dataframe - wrapper for run_complete_feature_engineering
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with raw data
        include_sentiment : bool
            Whether to include sentiment features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all engineered features
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Invalid DataFrame")
            return df
            
        logger.info(f"Adding all features to DataFrame with {len(df)} rows")
        
        # Step 1: Add technical indicators
        result = self.add_technical_indicators(df)
        
        # Step 2: Add price features
        result = self.add_price_features(result)
        
        # Step 3: Add sentiment features if requested
        if include_sentiment:
            result = self.add_sentiment_features(result)
        
        # Step 4: Add target variables
        result = self.add_target_variables(result)
        
        # Calculate how many features were added
        added_features = len(result.columns) - len(df.columns)
        logger.info(f"Added a total of {added_features} features")
        
        return result
    
    def prepare_ml_dataset(self, df, test_size=0.2, target_column='Target_Direction_1d'):
        """
        Prepare dataset for machine learning by splitting into train/test sets
        and handling missing values
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with engineered features
        test_size : float
            Proportion of data to use for testing
        target_column : str
            Name of the target column to predict
            
        Returns:
        --------
        dict
            Dictionary containing X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        if not isinstance(df, pd.DataFrame) or target_column not in df.columns:
            logger.error(f"Invalid DataFrame or missing target column '{target_column}'")
            return None
        
        # Copy DataFrame
        result_df = df.copy()
        
        # Remove rows with missing values in the target column
        result_df = result_df.dropna(subset=[target_column])
        
        # Identify date column if present
        date_column = None
        for col in ['Date', 'date', 'datetime', 'timestamp']:
            if col in result_df.columns:
                date_column = col
                break
        
        # Store dates if available
        dates = None
        if date_column:
            dates = result_df[date_column]
            result_df = result_df.drop(date_column, axis=1)
        
        # Remove non-feature columns
        non_feature_cols = [
            'ticker', 'Ticker', 'Symbol', 'symbol',  # Ticker symbols
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',  # Raw price data
            'Dividends', 'Stock Splits'  # Additional raw data
        ]
        
        # Also remove target columns (they start with 'Target_')
        target_cols = [col for col in result_df.columns if col.startswith('Target_')]
        
        # Create feature set by removing non-features and target
        feature_cols = [col for col in result_df.columns 
                      if col not in non_feature_cols + target_cols]
        
        # Extract features and target
        X = result_df[feature_cols]
        y = result_df[target_column]
        
        # Replace infinities with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column means
        X = X.fillna(X.mean())
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Time series data shouldn't be shuffled
        )
        
        # Calculate start and end dates of train/test if dates available
        if dates is not None:
            train_dates = dates.iloc[:len(X_train)]
            test_dates = dates.iloc[len(X_train):len(X_train) + len(X_test)]
            
            logger.info(f"Train set: {len(X_train)} samples from {train_dates.iloc[0]} to {train_dates.iloc[-1]}")
            logger.info(f"Test set: {len(X_test)} samples from {test_dates.iloc[0]} to {test_dates.iloc[-1]}")
        else:
            logger.info(f"Train set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'train_dates': train_dates if dates is not None else None,
            'test_dates': test_dates if dates is not None else None
        }