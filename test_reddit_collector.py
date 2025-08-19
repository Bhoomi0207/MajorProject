import os
import sys
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

print(f"Client ID: {os.environ.get('REDDIT_CLIENT_ID')}")
print(f"Client Secret: {'*' * len(os.environ.get('REDDIT_CLIENT_SECRET', '')) if os.environ.get('REDDIT_CLIENT_SECRET') else None}")
print(f"User Agent: {os.environ.get('REDDIT_USER_AGENT')}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import the RedditDataCollector class
sys.path.append(".")  # Make sure Python can find your modules
from src.data_collection.reddit_api import RedditDataCollector

def main():
    # Get credentials from environment variables
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")


    
    # Check if credentials are available
    if not all([client_id, client_secret, user_agent]):
        logger.error("Missing Reddit API credentials!")
        logger.info("Please set the following environment variables:")
        logger.info("REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        return
        
    logger.info("Initializing Reddit data collector...")
    try:
        reddit = RedditDataCollector(client_id, client_secret, user_agent)
        
        # Define stock symbols to fetch data for
        symbols = ["AAPL", "MSFT", "TSLA", "AMZN"]
        
        # Define subreddits to search
        subreddits = ["wallstreetbets", "stocks", "investing"]
        
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}...")
            
            # Try individual subreddits first
            for subreddit in subreddits:
                logger.info(f"Searching r/{subreddit} for {symbol}...")
                posts_df = reddit.fetch_subreddit_posts(
                    ticker=symbol,
                    subreddit_name=subreddit,
                    limit=5,
                    time_filter="month"
                )
                
                if posts_df is not None and not posts_df.empty:
                    logger.info(f"Found {len(posts_df)} posts for {symbol} in r/{subreddit}")
                    
                    # Display sample data
                    print(f"\nSample data for {symbol} from r/{subreddit}:")
                    sample = posts_df[["title", "score", "num_comments", "created_utc"]].head(2)
                    print(sample)
                else:
                    logger.warning(f"No posts found for {symbol} in r/{subreddit}")
            
            # Try combined search across multiple subreddits
            logger.info(f"Searching across multiple subreddits for {symbol}...")
            combined_df = reddit.fetch_multiple_subreddits(
                ticker=symbol,
                subreddits=subreddits,
                limit_per_subreddit=3
            )
            
            if combined_df is not None and not combined_df.empty:
                logger.info(f"Found total of {len(combined_df)} posts for {symbol} across all subreddits")
            else:
                logger.warning(f"No posts found for {symbol} across the subreddits")
            
            print("\n" + "-"*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error running Reddit data collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")