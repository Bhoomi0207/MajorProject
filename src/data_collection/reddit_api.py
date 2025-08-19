import praw
import pandas as pd
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataCollector:
    """
    Class to collect stock-related posts and comments from Reddit
    """
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize Reddit API with your credentials
        
        Get your credentials by creating a Reddit app at:
        https://www.reddit.com/prefs/apps
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        # Create data directory
        os.makedirs("data/reddit", exist_ok=True)
        
    def fetch_subreddit_posts(self, ticker, subreddit_name="wallstreetbets", 
                              limit=50, time_filter="week", save=True):
        """
        Fetch posts related to a specific ticker from a subreddit
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        subreddit_name : str
            Name of the subreddit
        limit : int
            Maximum number of posts to fetch
        time_filter : str
            Time filter for posts (hour, day, week, month, year, all)
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing posts data
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Search for posts containing the ticker
            search_query = f"{ticker}"
            posts = []
            
            logger.info(f"Fetching posts from r/{subreddit_name} for {ticker}...")
            
            # Search for posts containing the ticker
            for post in subreddit.search(search_query, sort="hot", time_filter=time_filter, limit=limit):
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'upvote_ratio': post.upvote_ratio,
                    'subreddit': subreddit_name,
                    'ticker': ticker
                })
            
            if not posts:
                logger.warning(f"No posts found for {ticker} in r/{subreddit_name}")
                return None
                
            df = pd.DataFrame(posts)
            logger.info(f"Successfully fetched {len(df)} posts for {ticker} from r/{subreddit_name}")
            
            if save:
                filename = f"data/reddit/{ticker}_{subreddit_name}_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved Reddit data to {filename}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return None
            
    def fetch_multiple_subreddits(self, ticker, subreddits=["wallstreetbets", "stocks", "investing"], 
                                 limit_per_subreddit=30):
        """
        Fetch posts related to a ticker from multiple subreddits
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        subreddits : list
            List of subreddit names
        limit_per_subreddit : int
            Maximum number of posts to fetch per subreddit
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame containing posts from all subreddits
        """
        all_posts = []
        
        for subreddit in subreddits:
            df = self.fetch_subreddit_posts(ticker, subreddit, limit_per_subreddit)
            if df is not None and not df.empty:
                all_posts.append(df)
                
        if not all_posts:
            logger.warning(f"No posts found for {ticker} in any subreddit")
            return None
            
        combined_df = pd.concat(all_posts, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} posts from {len(subreddits)} subreddits for {ticker}")
        
        return combined_df