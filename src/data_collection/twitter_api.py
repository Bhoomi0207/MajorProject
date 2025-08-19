import tweepy
import pandas as pd
from datetime import datetime
import time
import logging
import os
from config.config import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataCollector:
    """
    Class to collect tweets related to stocks from Twitter API (free tier)
    with rate limit handling and efficient usage
    """
    def __init__(self):
        self.authenticate()
        # Create data directory if it doesn't exist
        os.makedirs("data/tweets", exist_ok=True)

    def authenticate(self):
        """
        Authenticate with Twitter API
        """
        try:
            auth = tweepy.OAuth1UserHandler(
                TWITTER_API_KEY, 
                TWITTER_API_SECRET,
                TWITTER_ACCESS_TOKEN,
                TWITTER_ACCESS_SECRET
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            # Test the credentials
            self.api.verify_credentials()
            logger.info("Twitter API authentication successful")
        except Exception as e:
            logger.error(f"Twitter API authentication failed: {e}")
            self.api = None

    def search_tweets(self, query, count=100, lang="en", result_type="mixed"):
        """
        Search for tweets with given query
        
        Parameters:
        -----------
        query : str
            Search query (e.g., "$AAPL", "Tesla stock")
        count : int
            Number of tweets to fetch
        lang : str
            Language of tweets
        result_type: str
            Type of results to return: 'mixed', 'recent', or 'popular'
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing tweets or None if API is not authenticated
        """
        if not self.api:
            logger.error("Twitter API not authenticated")
            return None
        
        try:
            tweets = []
            # Use Cursor to handle pagination efficiently
            for tweet in tweepy.Cursor(
                self.api.search_tweets, 
                q=query,
                lang=lang,
                result_type=result_type,
                tweet_mode="extended",
                count=min(100, count)  # API limit per request
            ).items(count):
                
                # Extract the full text, handling retweets
                if hasattr(tweet, "retweeted_status"):
                    text = tweet.retweeted_status.full_text
                else:
                    text = tweet.full_text
                    
                tweet_data = {
                    "id": tweet.id,
                    "text": text,
                    "created_at": tweet.created_at,
                    "user": tweet.user.screen_name,
                    "followers_count": tweet.user.followers_count,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count,
                    "is_retweet": hasattr(tweet, "retweeted_status")
                }
                tweets.append(tweet_data)
                
            logger.info(f"Successfully fetched {len(tweets)} tweets for query: {query}")
            return pd.DataFrame(tweets)
            
        except tweepy.TweepyException as e:
            if "Rate limit" in str(e):
                logger.warning("Rate limit reached. Waiting to continue...")
                time.sleep(60 * 15)  # Wait for 15 minutes
                return self.search_tweets(query, count, lang)
            else:
                logger.error(f"Error fetching tweets: {e}")
                return None
    
    def fetch_stock_tweets(self, ticker_symbol, count=100, save=True):
        """
        Fetch tweets related to a specific stock ticker
        
        Parameters:
        -----------
        ticker_symbol : str
            Stock ticker symbol (e.g., "AAPL", "MSFT")
        count : int
            Number of tweets to fetch
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing tweets or None if API is not authenticated
        """
        # Search for cashtag and company mentions
        query = f"${ticker_symbol} OR {ticker_symbol} stock OR {ticker_symbol} price"
        df = self.search_tweets(query, count)
        
        if df is not None and not df.empty and save:
            # Save to CSV
            filename = f"data/tweets/{ticker_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved tweets to {filename}")
            
        return df
    
    def fetch_batch(self, ticker_list, count_per_ticker=50):
        """
        Fetch tweets for multiple stock tickers efficiently
        
        Parameters:
        -----------
        ticker_list : list
            List of ticker symbols
        count_per_ticker : int
            Number of tweets to fetch per ticker
            
        Returns:
        --------
        dict
            Dictionary with ticker symbols as keys and DataFrames as values
        """
        result = {}
        for ticker in ticker_list:
            logger.info(f"Fetching tweets for {ticker}...")
            df = self.fetch_stock_tweets(ticker, count_per_ticker)
            if df is not None and not df.empty:
                df['ticker'] = ticker
                result[ticker] = df
                
                # Implement a small delay between requests to avoid hitting rate limits
                if ticker != ticker_list[-1]:  # Don't delay after the last ticker
                    time.sleep(5)
                    
        return result
    
    def load_historical_data(self, ticker_symbol):
        """
        Load all historical tweet data for a ticker from saved CSV files
        
        Parameters:
        -----------
        ticker_symbol : str
            Stock ticker symbol
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame with all historical tweets for the ticker
        """
        files = [f for f in os.listdir("data/tweets") if f.startswith(f"{ticker_symbol}_")]
        if not files:
            logger.info(f"No historical data found for {ticker_symbol}")
            return None
            
        dfs = []
        for file in files:
            df = pd.read_csv(f"data/tweets/{file}")
            dfs.append(df)
            
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['id'])
            logger.info(f"Loaded {len(combined_df)} historical tweets for {ticker_symbol}")
            return combined_df
        
        return None