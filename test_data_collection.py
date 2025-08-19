import logging
import argparse
import sys
import os
import pandas as pd
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import data collection modules
try:
    from src.data_collection.yahoo_finance import YahooFinanceCollector
    from src.data_collection.stocktwits_api import StocktwitsDataCollector
    from src.data_collection.data_collector import DataCollector
    from src.data_collection.reddit_api import RedditDataCollector  # Import the Reddit collector
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def test_yahoo_finance():
    """Test Yahoo Finance data collection"""
    logger.info("Testing Yahoo Finance data collection...")
    
    try:
        yahoo = YahooFinanceCollector()
        
        # Test fetching stock data
        ticker = "AAPL"
        stock_data = yahoo.get_stock_data(ticker, save=True)
        
        if stock_data is None or stock_data.empty:
            logger.error(f"Failed to fetch Yahoo Finance data for {ticker}")
            return False
            
        logger.info(f"Successfully fetched {len(stock_data)} rows of stock data for {ticker}")
        
        # Check if 'close' exists in the columns (case-insensitive)
        close_column = None
        for col in stock_data.columns:
            if col.lower() == 'close':
                close_column = col
                break
        
        if close_column:
            logger.info(f"Latest close price: {stock_data.iloc[-1][close_column]}")
        else:
            logger.warning(f"Close column not found. Available columns: {list(stock_data.columns)}")
            return False
        
        # Test company info
        try:
            company_info = yahoo.get_company_info(ticker)
            if not company_info:
                logger.error(f"Failed to fetch company info for {ticker}")
                return False
                
            name_field = None
            for field in ['longName', 'shortName', 'name']:
                if field in company_info:
                    name_field = field
                    break
                    
            if name_field:
                logger.info(f"Company info for {ticker}: {company_info.get(name_field, '')}")
            else:
                logger.info(f"Company name not found. Available fields: {list(company_info.keys())[:5]}")
            
            return True
        except AttributeError:
            logger.warning("get_company_info method not found in YahooFinanceCollector class")
            # This is not a critical failure, so continue
            return True
        
    except Exception as e:
        logger.error(f"Error in Yahoo Finance test: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_stocktwits():
    """Test StockTwits data collection with fallback to alternative methods"""
    logger.info("Testing StockTwits API integration...")
    
    try:
        stocktwits = StocktwitsDataCollector()
        ticker = "AAPL"
        
        # First attempt: Try the standard API
        data = stocktwits.fetch_symbol_messages(ticker, limit=10, save=False)
        
        if data is not None and not data.empty:
            logger.info(f"Successfully fetched {len(data)} StockTwits messages for {ticker} via API")
            return True
            
        logger.warning("Standard API access failed. Trying alternative method...")
        
        # Second attempt: Try web scraping if available
        try:
            if hasattr(stocktwits, '_fetch_via_scraping'):
                scraped_data = stocktwits._fetch_via_scraping(ticker, limit=10, save=False)
                
                if scraped_data is not None and not scraped_data.empty:
                    logger.info(f"Successfully fetched {len(scraped_data)} StockTwits messages for {ticker} via scraping")
                    return True
                    
                logger.warning("Web scraping method also failed.")
        except AttributeError:
            logger.warning("Scraping method not available in StocktwitsDataCollector")
        
        # If both approaches fail, suggest alternatives
        logger.error("Failed to fetch StockTwits messages for AAPL")
        logger.info("Recommending: Use Reddit or Twitter as alternative sentiment data sources")
        
        # For testing purposes, we'll consider this a "soft failure" - return True if you want tests to pass
        return False
        
    except Exception as e:
        logger.error(f"Error in StockTwits test: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_news_scraper():
    """Test news scraper functionality"""
    logger.info("Testing news scraping functionality...")
    
    try:
        # Import news scraper if available
        try:
            from src.data_collection.news_scraper import NewsScraper
            news_scraper = NewsScraper()
        except (ImportError, ModuleNotFoundError):
            logger.warning("NewsScraper module not found. Skipping test.")
            return True
            
        ticker = "AAPL"
        news_data = news_scraper.fetch_yahoo_finance_news(ticker, save=False)
        
        if news_data is None or len(news_data) == 0:
            logger.warning(f"No news data found for {ticker} on Yahoo Finance")
            
            # Try another source
            news_data = news_scraper.fetch_seeking_alpha_news(ticker, save=False)
            
            if news_data is None or len(news_data) == 0:
                logger.warning(f"No news data found for {ticker} on Seeking Alpha")
                logger.error("Failed to fetch news data from any source")
                return False
        
        logger.info(f"Successfully fetched {len(news_data)} news articles for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error in news scraper test: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_data_collector():
    """Test overall data collector functionality"""
    logger.info("Testing DataCollector integration...")
    
    try:
        collector = DataCollector()
        ticker = "AAPL"
        
        # Test overall collection
        try:
            all_data = collector.collect_data_for_ticker(ticker)
            
            # Check if any data was collected
            if not all_data or all(data is None for data in all_data.values()):
                logger.warning(f"No data collected for {ticker}")
                return False
                
            # Count successful collections
            success_count = sum(1 for data in all_data.values() if data is not None)
            total_sources = len(all_data)
            
            logger.info(f"Successfully collected data from {success_count}/{total_sources} sources for {ticker}")
            
            # Consider test successful if at least one source provided data
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            if args.debug:
                logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        logger.error(f"Error in DataCollector test: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_reddit():
    """Test Reddit data collection using reddit_api.py"""
    logger.info("Testing Reddit data collection...")
    
    try:
        # Get Reddit API credentials from environment variables
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT", "python:stock-sentiment:v1.0 (by /u/YourUsername)")
        
        if not client_id or not client_secret:
            logger.warning("Reddit API credentials not found in environment variables.")
            logger.info("Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT environment variables.")
            return False
        
        # Initialize the Reddit collector
        reddit = RedditDataCollector(client_id, client_secret, user_agent)
        
        # Test single subreddit scraping
        ticker = "AAPL"
        logger.info(f"Testing single subreddit collection for {ticker}...")
        single_subreddit_data = reddit.fetch_subreddit_posts(
            ticker=ticker,
            subreddit_name="wallstreetbets",
            limit=10,
            save=False
        )
        
        if single_subreddit_data is None or single_subreddit_data.empty:
            logger.warning(f"No posts found for {ticker} in r/wallstreetbets")
        else:
            logger.info(f"Successfully fetched {len(single_subreddit_data)} posts for {ticker} from r/wallstreetbets")
        
        # Test multiple subreddit scraping
        logger.info(f"Testing multiple subreddits collection for {ticker}...")
        multi_subreddit_data = reddit.fetch_multiple_subreddits(
            ticker=ticker,
            subreddits=["wallstreetbets", "stocks", "investing"],
            limit_per_subreddit=10
        )
        
        if multi_subreddit_data is None or multi_subreddit_data.empty:
            logger.warning(f"No posts found for {ticker} across multiple subreddits")
            return False
        else:
            logger.info(f"Successfully fetched {len(multi_subreddit_data)} posts for {ticker} from multiple subreddits")
            
            # Display some data stats
            subreddit_counts = multi_subreddit_data['subreddit'].value_counts()
            logger.info(f"Posts by subreddit: {dict(subreddit_counts)}")
            
            if 'score' in multi_subreddit_data.columns:
                avg_score = multi_subreddit_data['score'].mean()
                logger.info(f"Average post score: {avg_score:.2f}")
            
            return True
        
    except Exception as e:
        logger.error(f"Error in Reddit test: {e}")
        if args.debug:
            logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_alternative_data():
    """Test alternative data sources if available"""
    logger.info("Testing alternative data sources...")
    
    # Test any available alternative sources
    success = False
    
    # Call the dedicated Reddit test function
    reddit_success = test_reddit()
    if reddit_success:
        success = True
    
    # Test News API if available
    try:
        from src.data_collection.news_api import NewsAPICollector
        logger.info("Testing News API integration...")
        
        # Check for API key
        api_key = os.environ.get("NEWS_API_KEY")
        if api_key:
            news_api = NewsAPICollector(api_key)
            ticker = "AAPL"
            news_data = news_api.fetch_stock_news(ticker, days_back=3, save=False)
            
            if news_data is not None and not news_data.empty:
                logger.info(f"Successfully fetched {len(news_data)} news articles for {ticker}")
                success = True
        else:
            logger.warning("News API key not found. Skipping News API test.")
    except (ImportError, ModuleNotFoundError):
        logger.info("News API module not available.")
    
    return success

def run_tests(modules):
    """Run specified test modules"""
    test_results = {}
    
    if not modules or 'yahoo' in modules:
        test_results['yahoo'] = test_yahoo_finance()
        
    if not modules or 'stocktwits' in modules:
        test_results['stocktwits'] = test_stocktwits()
        
    if not modules or 'scraper' in modules:
        test_results['scraper'] = test_news_scraper()
        
    if not modules or 'collector' in modules:
        test_results['collector'] = test_data_collector()
    
    # Add dedicated Reddit test
    if not modules or 'reddit' in modules:
        test_results['reddit'] = test_reddit()
        
    if not modules or 'alternative' in modules:
        test_results['alternative'] = test_alternative_data()
        
    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data collection modules")
    parser.add_argument("--module", "-m", 
                       help="Specific module to test (yahoo, stocktwits, scraper, collector, reddit, alternative)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    start_time = datetime.now()
    logger.info(f"Starting data collection tests at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    modules_to_test = [args.module] if args.module else None
    results = run_tests(modules_to_test)
    
    # Report results
    for module, passed in results.items():
        if passed:
            logger.info(f"Test PASSED for module: {module}")
        else:
            logger.info(f"Test FAILED for module: {module}")
    
    end_time = datetime.now()
    logger.info(f"Completed data collection tests at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")