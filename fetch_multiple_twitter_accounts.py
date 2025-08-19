import tweepy
import pandas as pd
from datetime import datetime
import time
import logging
import os
import random
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MultiAccountTwitterCollector:
    """
    Class to collect tweets using multiple Twitter API accounts to maximize API usage
    """
    def __init__(self, accounts_credentials):
        """
        Initialize with a list of account credentials
        
        Parameters:
        -----------
        accounts_credentials : list of dict
            List of dictionaries containing credentials for each account
            Each dict should have: api_key, api_secret, access_token, access_secret
        """
        self.accounts = []
        self.current_account_index = 0
        
        for creds in accounts_credentials:
            try:
                auth = tweepy.OAuth1UserHandler(
                    creds["api_key"],
                    creds["api_secret"],
                    creds["access_token"],
                    creds["access_secret"]
                )
                api = tweepy.API(auth, wait_on_rate_limit=True)
                
                # Test the credentials
                user = api.verify_credentials()
                logger.info(f"Successfully authenticated Twitter account: @{user.screen_name}")
                
                self.accounts.append({
                    "api": api,
                    "username": user.screen_name,
                    "usage_count": 0
                })
            except Exception as e:
                logger.error(f"Failed to authenticate Twitter account: {e}")
        
        # Create data directory if it doesn't exist
        os.makedirs("data/tweets", exist_ok=True)
        
        if not self.accounts:
            logger.error("No Twitter accounts were successfully authenticated")
            raise ValueError("No valid Twitter accounts provided")
            
        logger.info(f"Successfully authenticated {len(self.accounts)} Twitter accounts")
    
    def get_next_account(self):
        """Get the next available account in rotation"""
        if not self.accounts:
            return None
            
        account = self.accounts[self.current_account_index]
        self.current_account_index = (self.current_account_index + 1) % len(self.accounts)
        account["usage_count"] += 1
        return account
    
    def search_tweets(self, query, count=25, lang="en", result_type="mixed"):
        """
        Search for tweets with given query, using account rotation
        
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
            DataFrame containing tweets
        """
        if not self.accounts:
            logger.error("No authenticated Twitter accounts available")
            return None
        
        # Get the next account to use
        account = self.get_next_account()
        api = account["api"]
        account_username = account["username"]
        
        logger.info(f"Using account @{account_username} to fetch tweets for: {query}")
        
        try:
            tweets = []
            # Use Cursor to handle pagination efficiently
            for tweet in tweepy.Cursor(
                api.search_tweets, 
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
                    "is_retweet": hasattr(tweet, "retweeted_status"),
                    "source_account": account_username  # Track which account fetched this
                }
                tweets.append(tweet_data)
                
            logger.info(f"Successfully fetched {len(tweets)} tweets for query: {query}")
            return pd.DataFrame(tweets)
            
        except tweepy.TweepyException as e:
            if "Rate limit" in str(e):
                logger.warning(f"Rate limit reached for account @{account_username}. Trying next account...")
                # Mark this account as rate limited by increasing its usage count substantially
                account["usage_count"] += 100
                
                # If we've tried all accounts, wait for rate limit reset
                if all(acc["usage_count"] >= 100 for acc in self.accounts):
                    logger.warning("All accounts are rate limited. Waiting to continue...")
                    time.sleep(60 * 15)  # Wait for 15 minutes
                    # Reset usage counts
                    for acc in self.accounts:
                        acc["usage_count"] = 0
                        
                # Try again with next account
                return self.search_tweets(query, count, lang, result_type)
            else:
                logger.error(f"Error fetching tweets: {e}")
                return None
    
    def fetch_stock_tweets(self, ticker_symbol, count=25, save=True):
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
            DataFrame containing tweets
        """
        # Create a company name mapping for better search results
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google OR Alphabet',
            'AMZN': 'Amazon',
            'META': 'Meta OR Facebook',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan Chase',
            'V': 'Visa',
            'BAC': 'Bank of America',
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer',
            'UNH': 'UnitedHealth',
            'XOM': 'Exxon Mobil',
            'CVX': 'Chevron',
            'CAT': 'Caterpillar',
            'BA': 'Boeing',
            'DIS': 'Disney',
            'SBUX': 'Starbucks',
            'NKE': 'Nike'
        }
        
        # Get the company name if available
        company_name = company_names.get(ticker_symbol, ticker_symbol)
        
        # Search for cashtag, company name and stock-related terms
        query = f"(${ticker_symbol} OR {company_name}) (stock OR price OR shares OR market OR investing OR earnings)"
        
        df = self.search_tweets(query, count)
        
        if df is not None and not df.empty:
            # Add ticker column
            df['ticker'] = ticker_symbol
            
            if save:
                # Save to CSV
                filename = f"data/tweets/{ticker_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {len(df)} tweets to {filename}")
            
        return df
    
    def fetch_batch(self, ticker_list, count_per_ticker=25, shuffle_tickers=True):
        """
        Fetch tweets for multiple stock tickers efficiently, rotating accounts
        
        Parameters:
        -----------
        ticker_list : list
            List of ticker symbols
        count_per_ticker : int
            Number of tweets to fetch per ticker
        shuffle_tickers : bool
            Whether to randomize ticker order to minimize pattern detection
            
        Returns:
        --------
        dict
            Dictionary with ticker symbols as keys and DataFrames as values
        """
        result = {}
        total_tweets = 0
        
        # Optional shuffle of tickers to randomize requests
        if shuffle_tickers:
            random.shuffle(ticker_list)
        
        for i, ticker in enumerate(ticker_list):
            logger.info(f"Fetching tweets for {ticker} ({i+1}/{len(ticker_list)})...")
            
            df = self.fetch_stock_tweets(ticker, count_per_ticker)
            
            if df is not None and not df.empty:
                result[ticker] = df
                total_tweets += len(df)
                logger.info(f"Retrieved {len(df)} tweets for {ticker}")
            else:
                logger.warning(f"No tweets found for {ticker}")
                
            # Implement a small delay between requests to avoid suspicious patterns
            if i < len(ticker_list) - 1:  # Don't delay after the last ticker
                delay = random.uniform(1.0, 3.0)
                logger.info(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)
        
        logger.info(f"Completed batch fetch. Retrieved {total_tweets} tweets for {len(result)} tickers")
        return result
    
    def load_historical_data(self, ticker_symbol=None):
        """
        Load historical tweet data from saved CSV files
        
        Parameters:
        -----------
        ticker_symbol : str or None
            Stock ticker symbol (None to load all data)
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame with historical tweets
        """
        if ticker_symbol:
            files = [f for f in os.listdir("data/tweets") if f.startswith(f"{ticker_symbol}_")]
        else:
            files = [f for f in os.listdir("data/tweets") if f.endswith(".csv")]
            
        if not files:
            logger.info(f"No historical data found" + (f" for {ticker_symbol}" if ticker_symbol else ""))
            return None
            
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(f"data/tweets/{file}")
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading file {file}: {e}")
                
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['id'])
            logger.info(f"Loaded {len(combined_df)} historical tweets" + 
                       (f" for {ticker_symbol}" if ticker_symbol else ""))
            return combined_df
        
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch tweets for stock tickers using multiple Twitter accounts")
    parser.add_argument("--tickers", type=str, 
                      default="AAPL,AMZN,BA,BAC,CAT,CVX,DIS,GOOGL,JNJ,JPM,META,MSFT,NKE,PFE,SBUX,UNH,V,XOM",
                      help="Comma-separated list of stock tickers to fetch")
    parser.add_argument("--count", type=int, default=25,
                      help="Number of tweets to fetch per ticker")
    parser.add_argument("--save", action="store_true", default=True,
                      help="Save tweets to CSV files")
    parser.add_argument("--combine", action="store_true", default=True,
                      help="Combine all tweets into a single CSV file")
    return parser.parse_args()

def get_account_credentials():
    """Load all Twitter account credentials"""
    credentials = []
    
    # Account 1 (using environment variables or direct values)
    account1 = {
        "api_key": "dcGkl7abJanKFvA4qUJanBNu5",
        "api_secret": "0tNg6OmFKV3BwavUimRONIa80EOfQ7t2CXwfQCsP1VlPTSB5c4",
        "access_token": "896711506754060288-4CmiURYOdSFtdDwKYFUwYGikyZX8WWm",
        "access_secret": "kJLq4WuOU9tT9shBOnpZjQ2Imh46KcBQAOXEURPtk71Ej"
    }
    credentials.append(account1)
    
    # Account 2
    account2 = {
        "api_key": "zjREJTHwk3a9x1sWzJqoTm2go",
        "api_secret": "qpmhOFlzweae5nIfHXHgkllflnJPBEcaJjwib9JuL2zG0VDnxP",
        "access_token": "1864352346849071104-E6sQnHaMKyUMszt2wJbEsA9fOaf1ZZ",
        "access_secret": "MEWjCFCowtN1mbKDjIkv2qxRUkECUHD7ymUlkWh0DLi46"
    }
    credentials.append(account2)
    
    # Account 3
    account3 = {
        "api_key": "htu8rCgL1ppp4I0x6vTv6YcxY", 
        "api_secret": "tWHR5J0LkbN1NTlBhs1UkXjvZJQgJNpIlM0jALWOJCNXJJ6F2f",
        "access_token": "1900805211691905024-nAhlPMC5g0AgKNFCUmCvKpoOnNoDsk",
        "access_secret": "ORTxCTcWB2uKZogY9C3l7lp4f07kVY2nZqZj3fBcazAzg"
    }
    credentials.append(account3)
    
    # Account 4
    account4 = {
        "api_key": "871TdUpIrzYBKvyJHRh3Zjvk8",
        "api_secret": "N3g9U3T2ZIjOipp6h3foLE7hwJlEU6xCEFlN1aUq2pVWIwEuru",
        "access_token": "1815638825408061440-Utk2ePaxbVkBv9ru5NkINylG3A1M7H",
        "access_secret": "peIFdQtNKsurnckNIT7pOjZtSkaySS1FBmqeEyyH3YVbY"
    }
    credentials.append(account4)
    
    return credentials

def main():
    args = parse_arguments()
    
    # Parse ticker list
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    logger.info(f"Preparing to fetch tweets for {len(tickers)} tickers: {tickers}")
    
    # Get account credentials
    credentials = get_account_credentials()
    logger.info(f"Loaded credentials for {len(credentials)} Twitter accounts")
    
    try:
        # Initialize the collector with all accounts
        collector = MultiAccountTwitterCollector(credentials)
        
        # Fetch tweets for all tickers
        results = collector.fetch_batch(tickers, count_per_ticker=args.count)
        
        # Calculate statistics
        total_tweets = sum(len(df) for df in results.values() if df is not None)
        tickers_with_data = sum(1 for df in results.values() if df is not None and not df.empty)
        
        # Print summary
        logger.info("\n===== SUMMARY =====")
        logger.info(f"Total tickers processed: {len(tickers)}")
        logger.info(f"Tickers with data: {tickers_with_data}")
        logger.info(f"Total tweets collected: {total_tweets}")
        
        if results:
            logger.info("\nBreakdown by ticker:")
            for ticker, df in results.items():
                if df is not None and not df.empty:
                    logger.info(f"  {ticker}: {len(df)} tweets")
                else:
                    logger.info(f"  {ticker}: No tweets found")
        
        if args.combine and total_tweets > 0:
            # Combine all data into a single CSV
            all_dfs = [df for df in results.values() if df is not None and not df.empty]
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                combined_filename = f"data/tweets/all_tickers_{timestamp}.csv"
                combined_df.to_csv(combined_filename, index=False)
                logger.info(f"Combined {len(combined_df)} tweets into: {combined_filename}")
                
                # Print the distribution of tweets by account
                if "source_account" in combined_df.columns:
                    account_counts = combined_df["source_account"].value_counts()
                    logger.info("\nDistribution by Twitter account:")
                    for account, count in account_counts.items():
                        logger.info(f"  @{account}: {count} tweets")
        
    except Exception as e:
        logger.error(f"Error running Twitter data collection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    exit_code = main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")
    
    sys.exit(exit_code)