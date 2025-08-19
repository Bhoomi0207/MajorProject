import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YahooFinanceCollector:
    """
    Class to collect stock data from Yahoo Finance
    """
    def __init__(self):
        # Create data directory if it doesn't exist
        os.makedirs("data/stocks", exist_ok=True)
    
    def get_stock_data(self, ticker, period="1y", interval="1d", save=True):
        """
        Get historical stock data
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "MSFT")
        period : str
            Time period to fetch data for (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
        interval : str
            Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing stock data or None if request fails
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None
                
            # Reset index to make Date a column
            hist = hist.reset_index()
            
            # Calculate additional technical indicators
            if len(hist) > 14:  # Need enough data points for indicators
                # Calculate daily returns
                hist['return'] = hist['Close'].pct_change() * 100
                
                # Calculate moving averages
                hist['MA5'] = hist['Close'].rolling(window=5).mean()
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                
                # Calculate RSI (Relative Strength Index)
                delta = hist['Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                hist['RSI'] = 100 - (100 / (1 + rs))
                
                # Calculate trading volume changes
                hist['volume_change'] = hist['Volume'].pct_change() * 100
            
            if save:
                # Save to CSV
                filename = f"data/stocks/{ticker}_data.csv"
                hist.to_csv(filename, index=False)
                logger.info(f"Saved stock data to {filename}")
            
            logger.info(f"Successfully fetched stock data for {ticker}")
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    def get_multiple_stocks_data(self, tickers, period="1y", interval="1d", save=True):
        """
        Get historical data for multiple stocks
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        period : str
            Time period to fetch data for
        interval : str
            Data interval
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        dict
            Dictionary with ticker symbols as keys and DataFrames as values
        """
        result = {}
        for ticker in tickers:
            df = self.get_stock_data(ticker, period, interval, save)
            if df is not None and not df.empty:
                result[ticker] = df
        return result
    
    def get_company_info(self, ticker):
        """
        Get company information
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        dict
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract the most important information
            important_info = {
                'shortName': info.get('shortName', ''),
                'longName': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'website': info.get('website', ''),
                'market': info.get('market', ''),
                'marketCap': info.get('marketCap', 0),
                'forwardPE': info.get('forwardPE', 0),
                'dividendYield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
                'averageVolume': info.get('averageVolume', 0),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0)
            }
            
            logger.info(f"Successfully fetched company info for {ticker}")
            return important_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return {}