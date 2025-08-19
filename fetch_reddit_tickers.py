import os
import sys
import pandas as pd
import logging
from datetime import datetime
import time
import argparse
import random
from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import the RedditDataCollector
try:
    from src.data_collection.reddit_api import RedditDataCollector
except ImportError:
    logger.error("RedditDataCollector module not found. Make sure src/data_collection/reddit_api.py exists")
    sys.exit(1)

def fetch_ticker_data(client_id, client_secret, user_agent, tickers, 
                     subreddits=None, posts_per_ticker=10, 
                     time_filter="month", add_delay=True):
    """
    Fetch Reddit data for multiple tickers
    
    Parameters:
    -----------
    client_id : str
        Reddit API client ID
    client_secret : str
        Reddit API client secret
    user_agent : str
        Reddit API user agent
    tickers : list
        List of ticker symbols to fetch data for
    subreddits : list
        List of subreddits to search in (default: ["wallstreetbets", "stocks", "investing"])
    posts_per_ticker : int
        Number of posts to fetch per ticker from each subreddit
    time_filter : str
        Time filter for Reddit search
    add_delay : bool
        Whether to add random delays between API calls
    
    Returns:
    --------
    dict
        Dictionary mapping tickers to their data
    """
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing"]
    
    # Initialize Reddit collector
    try:
        reddit = RedditDataCollector(client_id, client_secret, user_agent)
    except Exception as e:
        logger.error(f"Failed to initialize Reddit collector: {e}")
        return {}
    
    results = {}
    total_requests = 0
    
    # Set maximum requests to avoid rate limiting
    MAX_REQUESTS = 60  # Conservative limit to avoid hitting Reddit's rate limits
    
    for i, ticker in enumerate(tickers):
        logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})...")
        
        try:
            # Check if we're approaching request limits
            if total_requests >= MAX_REQUESTS:
                logger.warning(f"Reached maximum request limit ({MAX_REQUESTS}). Stopping to avoid rate limiting.")
                break
            
            # For popular tickers like AAPL, use fewer subreddits to avoid excessive data
            ticker_subreddits = subreddits
            if ticker in ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]:
                # For very popular tickers, maybe use fewer subreddits or limit posts further
                posts_limit = max(5, posts_per_ticker - 2)  # Slightly reduce post count for popular tickers
            else:
                posts_limit = posts_per_ticker
            
            # Fetch data from multiple subreddits
            df = reddit.fetch_multiple_subreddits(
                ticker=ticker,
                subreddits=ticker_subreddits,
                limit_per_subreddit=posts_limit
            )
            
            # Count this as one request per subreddit
            total_requests += len(ticker_subreddits)
            
            if df is not None and not df.empty:
                logger.info(f"Found {len(df)} posts for {ticker}")
                results[ticker] = df
                
                # Save the data
                output_dir = "data/reddit"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{ticker}_reddit_data_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {ticker} data to {output_file}")
            else:
                logger.warning(f"No data found for {ticker}")
                results[ticker] = None
            
            # Add a random delay between requests to avoid rate limiting
            if add_delay and i < len(tickers) - 1:
                delay = random.uniform(2.0, 5.0)
                logger.info(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            results[ticker] = None
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Fetch Reddit data for stock tickers")
    parser.add_argument("--tickers", "-t", type=str, 
                       default="AAPL,AMZN,BA,BAC,CAT,CVX,DIS,GOOGL,JNJ,JPM,META,MSFT,NKE,PFE,SBUX,UNH,V,XOM",
                       help="Comma-separated list of tickers to fetch data for")
    parser.add_argument("--subreddits", "-s", type=str, 
                       default="wallstreetbets,stocks,investing",
                       help="Comma-separated list of subreddits to search in")
    parser.add_argument("--posts", "-p", type=int, default=8,
                       help="Number of posts to fetch per ticker from each subreddit")
    parser.add_argument("--time-filter", "-tf", type=str, default="month",
                       choices=["day", "week", "month", "year", "all"],
                       help="Time filter for Reddit search")
    parser.add_argument("--no-delay", action="store_true",
                       help="Disable random delays between API calls")
    
    args = parser.parse_args()
    
    # Get Reddit API credentials from environment variables
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")
    
    if not client_id or not client_secret or not user_agent:
        logger.error("Reddit API credentials not found in environment variables")
        logger.info("Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
        sys.exit(1)
    
    # Parse tickers and subreddits
    tickers = [t.strip() for t in args.tickers.split(",")]
    subreddits = [s.strip() for s in args.subreddits.split(",")]
    
    logger.info(f"Fetching data for {len(tickers)} tickers from {len(subreddits)} subreddits")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Subreddits: {subreddits}")
    
    # Fetch data
    results = fetch_ticker_data(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        tickers=tickers,
        subreddits=subreddits,
        posts_per_ticker=args.posts,
        time_filter=args.time_filter,
        add_delay=not args.no_delay
    )
    
    # Report summary
    total_posts = sum(len(df) for df in results.values() if df is not None)
    successful_tickers = sum(1 for df in results.values() if df is not None)
    
    logger.info("\n===== SUMMARY =====")
    logger.info(f"Tickers processed: {len(tickers)}")
    logger.info(f"Tickers with data: {successful_tickers}")
    logger.info(f"Total posts collected: {total_posts}")
    
    if successful_tickers > 0:
        logger.info("\nTicker statistics:")
        for ticker, df in results.items():
            if df is not None:
                logger.info(f"  {ticker}: {len(df)} posts")
    
    logger.info("====================")
    
if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")