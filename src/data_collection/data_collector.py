import logging
import pandas as pd
import os
from datetime import datetime
import time
from config.config import STOCKS_TO_TRACK, SECTOR_MAPPING
from src.data_collection.twitter_api import TwitterDataCollector
from src.data_collection.stocktwits_api import StocktwitsDataCollector
from src.data_collection.yahoo_finance import YahooFinanceCollector
from src.data_collection.web_scraper import NewsScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Main class to collect and organize data from all sources
    """
    def __init__(self):
        # Initialize individual collectors
        self.twitter = TwitterDataCollector()
        self.stocktwits = StocktwitsDataCollector()
        self.yahoo = YahooFinanceCollector()
        self.news = NewsScraper()
        
        # Create main data directory
        os.makedirs("data", exist_ok=True)
        
    def collect_data_for_ticker(self, ticker, twitter_count=100):
        """
        Collect all available data for a single ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        twitter_count : int
            Number of tweets to fetch
            
        Returns:
        --------
        dict
            Dictionary with data sources as keys and DataFrames as values
        """
        logger.info(f"Collecting data for {ticker}")
        result = {}
        
        # 1. Stock data from Yahoo Finance
        stock_data = self.yahoo.get_stock_data(ticker)
        if stock_data is not None:
            result['stock_data'] = stock_data
            
        # 2. Company info
        company_info = self.yahoo.get_company_info(ticker)
        if company_info:
            result['company_info'] = company_info
            
        # 3. Twitter data
        twitter_data = self.twitter.fetch_stock_tweets(ticker, twitter_count)
        if twitter_data is not None:
            result['twitter'] = twitter_data
            
        # 4. StockTwits data
        stocktwits_data = self.stocktwits.fetch_symbol_messages(ticker)
        if stocktwits_data is not None:
            result['stocktwits'] = stocktwits_data
            
        # 5. News data from multiple sources
        news_data = self.news.scrape_all_sources(ticker)
        if news_data:
            result['news'] = news_data
            
        logger.info(f"Completed data collection for {ticker}")
        return result
    
    def collect_data_batch(self, tickers=None, delay_between_tickers=5):
        """
        Collect data for multiple tickers
        
        Parameters:
        -----------
        tickers : list or None
            List of ticker symbols. If None, use default tickers from config
        delay_between_tickers : int
            Delay between processing tickers in seconds
            
        Returns:
        --------
        dict
            Dictionary with tickers as keys and collected data as values
        """
        if tickers is None:
            tickers = STOCKS_TO_TRACK
            
        logger.info(f"Starting batch collection for {len(tickers)} tickers")
        
        results = {}
        for i, ticker in enumerate(tickers):
            logger.info(f"Processing ticker {i+1}/{len(tickers)}: {ticker}")
            results[ticker] = self.collect_data_for_ticker(ticker)
            
            # Add delay between tickers to avoid rate limits
            if i < len(tickers) - 1:  # Don't delay after the last ticker
                logger.info(f"Waiting {delay_between_tickers} seconds before next ticker...")
                time.sleep(delay_between_tickers)
                
        logger.info(f"Completed batch collection for {len(tickers)} tickers")
        return results
    
    def combine_sentiment_data(self, ticker):
        """
        Combine sentiment data from multiple sources for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame with sentiment data
        """
        dfs = []
        
        # 1. Twitter historical data
        twitter_df = self.twitter.load_historical_data(ticker)
        if twitter_df is not None:
            twitter_df['source'] = 'Twitter'
            twitter_df = twitter_df[['text', 'created_at', 'source', 'ticker']]
            dfs.append(twitter_df)
            
        # 2. StockTwits historical data
        stocktwits_df = self.stocktwits.load_historical_data(ticker)
        if stocktwits_df is not None:
            stocktwits_df['source'] = 'StockTwits'
            stocktwits_df = stocktwits_df[['text', 'created_at', 'source', 'sentiment', 'symbol']]
            stocktwits_df = stocktwits_df.rename(columns={'symbol': 'ticker'})
            dfs.append(stocktwits_df)
            
        # Combine data if available
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} sentiment data points for {ticker}")
            return combined_df
        else:
            logger.warning(f"No sentiment data found for {ticker}")
            return None
    
    def get_sector_data(self, sector):
        """
        Get data for all tickers in a specific sector
        
        Parameters:
        -----------
        sector : str
            Sector name
            
        Returns:
        --------
        dict
            Dictionary with tickers as keys and collected data as values
        """
        # Filter tickers by sector
        sector_tickers = [ticker for ticker, mapped_sector in SECTOR_MAPPING.items() 
                         if mapped_sector == sector]
        
        if not sector_tickers:
            logger.warning(f"No tickers found for sector: {sector}")
            return {}
            
        logger.info(f"Collecting data for {len(sector_tickers)} tickers in {sector} sector")
        return self.collect_data_batch(sector_tickers)