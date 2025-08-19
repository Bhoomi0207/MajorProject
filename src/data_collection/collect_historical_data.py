"""
Script to collect historical data for stock sentiment analysis
"""
import os
import sys
import logging
import argparse
from datetime import datetime
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_collection.historical_data import HistoricalDataCollector, load_historical_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Collect historical data for stock sentiment analysis")
    parser.add_argument("--tickers", type=str, 
                      default="AAPL,MSFT,GOOGL,AMZN,META,JPM,V",
                      help="Comma-separated list of stock tickers to fetch")
    parser.add_argument("--days", type=int, default=180,
                      help="Number of days of historical data to collect")
    parser.add_argument("--no-news", action="store_true",
                      help="Skip news data collection (to avoid API limits)")
    parser.add_argument("--subreddits", type=str,
                      default="wallstreetbets,stocks,investing",
                      help="Comma-separated list of subreddits to search")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Parse ticker list
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    logger.info(f"Preparing to fetch historical data for {len(tickers)} tickers: {tickers}")
    
    # Parse subreddit list
    subreddits = [subreddit.strip() for subreddit in args.subreddits.split(",")]
    
    # Initialize the collector
    collector = HistoricalDataCollector()
    
    # Process each ticker
    results = {}
    for i, ticker in enumerate(tickers):
        logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})...")
        
        # Fetch price data
        price_data = collector.get_historical_prices(ticker, period="1y")
        
        # Fetch Reddit data
        reddit_data = []
        for subreddit in subreddits:
            logger.info(f"Fetching Reddit data for {ticker} from r/{subreddit}...")
            data = collector.get_historical_reddit(ticker, subreddit=subreddit, days_back=args.days)
            if data is not None:
                reddit_data.append(data)
            
            # Add a small delay to avoid API rate limits
            time.sleep(2)
        
        # Fetch news data if requested
        if not args.no_news:
            news_data = collector.get_historical_news(ticker, days_back=min(30, args.days))
        
        # Create merged dataset
        merged_data = collector.create_merged_dataset(ticker, lookback_days=args.days)
        results[ticker] = merged_data
        
        # Add a delay between tickers
        if i < len(tickers) - 1:
            delay = 5
            logger.info(f"Waiting {delay} seconds before next ticker...")
            time.sleep(delay)
    
    # Print summary
    logger.info("\n===== SUMMARY =====")
    logger.info(f"Processed {len(tickers)} tickers")
    successful = sum(1 for data in results.values() if data is not None)
    logger.info(f"Successfully collected data for {successful} tickers")
    
    for ticker, data in results.items():
        if data is not None:
            logger.info(f"  {ticker}: {len(data)} days of historical data")
        else:
            logger.info(f"  {ticker}: Failed to collect data")
    
    logger.info("====================")
    logger.info(f"Historical data saved to {os.path.abspath(collector.hist_dir)}")
    
if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")