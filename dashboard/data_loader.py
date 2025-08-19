"""
Stock Data Loader for Dashboard Application
Handles loading and preprocessing of stock data for visualization
"""
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class StockDataLoader:
    """
    Class to load and preprocess stock data for dashboard visualization
    """
    def __init__(self, data_directory="data/historical", news_directory="data/news"):
        """
        Initialize the stock data loader
        
        Parameters:
        -----------
        data_directory : str
            Directory containing historical stock data
        news_directory : str
            Directory containing news data
        """
        self.data_directory = data_directory
        self.news_directory = news_directory
        self.available_stocks = self._get_available_stocks()
        self.loaded_data = {}
        self.loaded_news = {}
    
    def _get_available_stocks(self):
        """
        Get list of available stock tickers from data directory
        
        Returns:
        --------
        list
            List of available stock tickers
        """
        available_stocks = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_directory):
            logger.warning(f"Data directory {self.data_directory} does not exist")
            return available_stocks
        
        # Look for CSV files in data directory
        try:
            for file in os.listdir(self.data_directory):
                if file.endswith("_merged_historical.csv"):
                    ticker = file.split("_")[0]
                    available_stocks.append(ticker)
            
            available_stocks.sort()
            logger.info(f"Found {len(available_stocks)} available stocks: {', '.join(available_stocks)}")
            return available_stocks
        
        except Exception as e:
            logger.error(f"Error getting available stocks: {e}")
            return []
    
    def load_stock_data(self, ticker, force_reload=False):
        """
        Load historical stock data for a given ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        force_reload : bool
            Whether to force reload if data is already loaded
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with historical stock data
        """
        # Check if already loaded
        if ticker in self.loaded_data and not force_reload:
            return self.loaded_data[ticker]
        
        # Check if ticker is available
        if ticker not in self.available_stocks:
            logger.warning(f"Ticker {ticker} is not available")
            return None
        
        # Load data
        file_path = os.path.join(self.data_directory, f"{ticker}_merged_historical.csv")
        try:
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Calculate additional metrics for visualization
            if 'Close' in df.columns:
                # Add daily returns
                df['Returns'] = df['Close'].pct_change()
                
                # Add moving averages
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # Add Bollinger Bands
                df['BB_middle'] = df['Close'].rolling(window=20).mean()
                df['BB_std'] = df['Close'].rolling(window=20).std()
                df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
                df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
                
                # Add volatility (rolling std of returns)
                df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            logger.info(f"Loaded data for {ticker} with {len(df)} rows")
            
            # Store data
            self.loaded_data[ticker] = df
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None
    
    def load_news_data(self, ticker, force_reload=False):
        """
        Load news data for a given ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        force_reload : bool
            Whether to force reload if data is already loaded
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with news data
        """
        # Check if already loaded
        if ticker in self.loaded_news and not force_reload:
            return self.loaded_news[ticker]
        
        # Look for news file
        news_files = []
        if os.path.exists(self.news_directory):
            for file in os.listdir(self.news_directory):
                if file.startswith(f"{ticker}_news_") and file.endswith(".csv"):
                    news_files.append(os.path.join(self.news_directory, file))
        
        if not news_files:
            logger.warning(f"No news data found for {ticker}")
            return None
        
        # Use the most recent news file
        latest_news_file = sorted(news_files)[-1]
        
        # Load news data
        try:
            news_df = pd.read_csv(latest_news_file)
            
            # Convert date column to datetime
            date_cols = [col for col in news_df.columns if 'date' in col.lower() or 'published' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                news_df[date_col] = pd.to_datetime(news_df[date_col])
                news_df.sort_values(by=date_col, ascending=False, inplace=True)
            
            logger.info(f"Loaded news data for {ticker} with {len(news_df)} articles from {latest_news_file}")
            
            # Store data
            self.loaded_news[ticker] = news_df
            
            return news_df
        
        except Exception as e:
            logger.error(f"Error loading news data for {ticker}: {e}")
            return None
    
    def get_price_range(self, ticker):
        """
        Get price range for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        tuple
            (min_price, max_price)
        """
        if ticker not in self.loaded_data:
            self.load_stock_data(ticker)
        
        if ticker in self.loaded_data and 'Close' in self.loaded_data[ticker].columns:
            min_price = self.loaded_data[ticker]['Close'].min()
            max_price = self.loaded_data[ticker]['Close'].max()
            return (min_price, max_price)
        
        return (0, 100)  # Default range
    
    def get_date_range(self, ticker):
        """
        Get date range for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        tuple
            (start_date, end_date)
        """
        if ticker not in self.loaded_data:
            self.load_stock_data(ticker)
        
        if ticker in self.loaded_data:
            start_date = self.loaded_data[ticker].index.min()
            end_date = self.loaded_data[ticker].index.max()
            return (start_date, end_date)
        
        return (datetime.now() - timedelta(days=365), datetime.now())  # Default range
    
    def filter_date_range(self, ticker, start_date, end_date):
        """
        Filter data for a specific date range
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        pandas.DataFrame
            Filtered DataFrame
        """
        if ticker not in self.loaded_data:
            self.load_stock_data(ticker)
        
        if ticker in self.loaded_data:
            return self.loaded_data[ticker].loc[start_date:end_date]
        
        return None