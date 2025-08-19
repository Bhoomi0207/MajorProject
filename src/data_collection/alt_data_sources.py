import pandas as pd
import numpy as np
import requests
import praw
import logging
import os
import json
from datetime import datetime
from bs4 import BeautifulSoup
from config.config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_FETCH_LIMIT,
    KAGGLE_STOCK_SENTIMENT_PATH,
    FINANCIAL_PHRASEBANK_PATH,
    DATA_PATH
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataCollector:
    """
    Class to collect posts from Reddit related to stocks
    Reddit is a good alternative source for stock sentiment
    """
    def __init__(self):
        self.authenticate()
        
        # Create cache directory if it doesn't exist
        self.cache_dir = os.path.join(DATA_PATH, "reddit_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def authenticate(self):
        """
        Authenticate with Reddit API
        """
        try:
            if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
                logger.warning("Reddit API credentials not found in environment variables")
                self.reddit = None
                return
                
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            logger.info("Reddit API authentication successful")
        except Exception as e:
            logger.error(f"Reddit API authentication failed: {e}")
            self.reddit = None
    
    def fetch_subreddit_posts(self, subreddit_name, limit=REDDIT_FETCH_LIMIT, sort='hot'):
        """
        Fetch posts from a specific subreddit
        
        Parameters:
        -----------
        subreddit_name : str
            Name of the subreddit (e.g., "wallstreetbets", "stocks", "investing")
        limit : int
            Maximum number of posts to fetch
        sort : str
            Sorting method ('hot', 'new', 'top', 'rising')
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing posts or None if API is not authenticated
        """
        if not self.reddit:
            logger.error("Reddit API not authenticated")
            return None
            
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{subreddit_name}_{sort}.csv")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            posts = []
            
            # Get posts based on sort method
            if sort == 'hot':
                submissions = subreddit.hot(limit=limit)
            elif sort == 'new':
                submissions = subreddit.new(limit=limit)
            elif sort == 'top':
                submissions = subreddit.top(limit=limit, time_filter='day')
            elif sort == 'rising':
                submissions = subreddit.rising(limit=limit)
            else:
                submissions = subreddit.hot(limit=limit)
                
            for submission in submissions:
                # Skip stickied posts
                if submission.stickied:
                    continue
                    
                post_data = {
                    "id": submission.id,
                    "title": submission.title,
                    "text": submission.selftext,
                    "created_at": datetime.fromtimestamp(submission.created_utc),
                    "author": str(submission.author),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": submission.upvote_ratio
                }
                posts.append(post_data)
                
            logger.info(f"Successfully fetched {len(posts)} posts from r/{subreddit_name}")
            df = pd.DataFrame(posts)
            
            # Save to cache
            if not df.empty:
                df.to_csv(cache_file, index=False)
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Reddit posts: {e}")
            # If we encountered an error, try to use cached data if available
            if os.path.exists(cache_file):
                logger.info(f"Using cached Reddit data for r/{subreddit_name}")
                return pd.read_csv(cache_file)
            return None
    
    def search_stock_mentions(self, ticker):
        """
        Search for mentions of a stock ticker across financial subreddits
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "MSFT")
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing relevant posts
        """
        financial_subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket"]
        all_posts = []
        
        for subreddit in financial_subreddits:
            df = self.fetch_subreddit_posts(subreddit)
            if df is not None and not df.empty:
                # Filter posts containing the ticker
                ticker_pattern = r'\b' + ticker + r'\b'
                mask = (df['title'].str.contains(ticker_pattern, case=False, na=False) | 
                        df['text'].str.contains(ticker_pattern, case=False, na=False))
                filtered_df = df[mask].copy()
                
                if not filtered_df.empty:
                    filtered_df['subreddit'] = subreddit
                    all_posts.append(filtered_df)
        
        if all_posts:
            return pd.concat(all_posts, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_stock_sentiment(self, ticker):
        """
        Get sentiment for a specific stock ticker from Reddit
        This is a simplified version - the actual sentiment analysis
        will be handled by the sentiment analysis module
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with posts mentioning the ticker
        """
        return self.search_stock_mentions(ticker)


class PretrainedDataLoader:
    """
    Class to load pre-downloaded financial sentiment datasets
    These can be used when API access is limited or for demonstrations
    """
    
    @staticmethod
    def load_kaggle_financial_sentiment():
        """
        Load the Kaggle Financial Sentiment dataset
        This assumes you've downloaded it from Kaggle and placed it in your data directory
        
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing pre-labeled financial sentiment data
        """
        if os.path.exists(KAGGLE_STOCK_SENTIMENT_PATH):
            try:
                df = pd.read_csv(KAGGLE_STOCK_SENTIMENT_PATH)
                logger.info(f"Successfully loaded Kaggle financial sentiment dataset with {len(df)} entries")
                return df
            except Exception as e:
                logger.error(f"Error loading Kaggle dataset: {e}")
                return None
        else:
            logger.warning(f"Kaggle dataset not found at {KAGGLE_STOCK_SENTIMENT_PATH}")
            return None
    
    @staticmethod
    def load_financial_phrasebank():
        """
        Load the Financial PhraseBank dataset
        This is a publicly available dataset for financial sentiment analysis
        
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing financial phrases and sentiment labels
        """
        if os.path.exists(FINANCIAL_PHRASEBANK_PATH):
            try:
                df = pd.read_csv(FINANCIAL_PHRASEBANK_PATH)
                logger.info(f"Successfully loaded Financial PhraseBank dataset with {len(df)} entries")
                return df
            except Exception as e:
                logger.error(f"Error loading Financial PhraseBank dataset: {e}")
                return None
        else:
            logger.warning(f"Financial PhraseBank dataset not found at {FINANCIAL_PHRASEBANK_PATH}")
            return None
    
    @staticmethod
    def generate_synthetic_data(n_samples=1000, tickers=None):
        """
        Generate synthetic financial sentiment data for demonstration
        This is useful when you have no API access or pre-downloaded datasets
        
        Parameters:
        -----------
        n_samples : int
            Number of synthetic samples to generate
        tickers : list
            List of stock tickers to use
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing synthetic financial sentiment data
        """
        if tickers is None:
            from config.config import DEFAULT_TICKERS
            tickers = DEFAULT_TICKERS
            
        # Sample texts for different sentiment categories
        positive_templates = [
            "{ticker} just announced fantastic earnings, exceeding expectations.",
            "Bullish on {ticker}, their new product looks promising.",
            "{ticker} stock is soaring after positive analyst coverage.",
            "Just bought more {ticker} shares, the future looks bright.",
            "The {ticker} CEO's strategy is working, revenues are up.",
            "{ticker} is outperforming the sector, strong buy recommendation.",
            "Impressed with {ticker}'s growth numbers this quarter."
        ]
        
        negative_templates = [
            "{ticker} missed earnings expectations, stock plummeting.",
            "Bearish on {ticker}, their market share is declining.",
            "{ticker} facing serious competition, concerned about outlook.",
            "Sold my {ticker} shares, too much uncertainty ahead.",
            "The {ticker} restructuring plan doesn't address core issues.",
            "{ticker} is overvalued compared to peers, expecting correction.",
            "{ticker}'s latest product launch was disappointing."
        ]
        
        neutral_templates = [
            "Watching {ticker} closely, waiting for more data.",
            "{ticker} announced a partnership, impact unclear yet.",
            "Anyone have thoughts on {ticker}'s recent price movement?",
            "Holding my {ticker} position for now, monitoring situation.",
            "{ticker} trading sideways after recent news.",
            "Considering adding {ticker} to my portfolio, need more research.",
            "What's everyone's take on {ticker} for long-term investment?"
        ]
        
        # Generate synthetic data
        data = []
        sentiments = ['positive', 'negative', 'neutral']
        sentiment_weights = [0.4, 0.4, 0.2]  # Biasing toward non-neutral for clearer patterns
        
        for i in range(n_samples):
            ticker = np.random.choice(tickers)
            sentiment = np.random.choice(sentiments, p=sentiment_weights)
            
            if sentiment == 'positive':
                text = np.random.choice(positive_templates).format(ticker=ticker)
            elif sentiment == 'negative':
                text = np.random.choice(negative_templates).format(ticker=ticker)
            else:
                text = np.random.choice(neutral_templates).format(ticker=ticker)
                
            # Add some random timestamp within last 30 days
            days_ago = np.random.randint(0, 30)
            hours_ago = np.random.randint(0, 24)
            created_at = datetime.now() - pd.Timedelta(days=days_ago, hours=hours_ago)
            
            data.append({
                'id': f"synthetic_{i}",
                'text': text,
                'created_at': created_at,
                'ticker': ticker,
                'sentiment': sentiment,
                'source': 'synthetic'
            })
            
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic financial sentiment records")
        
        # Save the synthetic data for future use
        synthetic_path = os.path.join(DATA_PATH, "synthetic_sentiment.csv")
        df.to_csv(synthetic_path, index=False)
        
        return df


class FinancialNewsCollector:
    """
    Class to scrape financial news headlines from free sources
    """
    
    def __init__(self):
        self.cache_dir = os.path.join(DATA_PATH, "news_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # User agent to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_yahoo_finance_news(self, ticker):
        """
        Scrape news headlines from Yahoo Finance for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing news headlines
        """
        # Cache file path
        cache_file = os.path.join(self.cache_dir, f"{ticker}_yahoo_news.csv")
        
        # Check if we have recent cached data (less than 6 hours old)
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < pd.Timedelta(hours=6):
                logger.info(f"Loading cached Yahoo Finance news for {ticker}")
                return pd.read_csv(cache_file)
        
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news headlines
            # Note: Yahoo Finance's structure might change, requiring adjustments
            news_elements = soup.select('div#quoteNewsStream li a')
            
            for element in news_elements:
                headline = element.get_text().strip()
                if headline:
                    news_items.append({
                        'headline': headline,
                        'ticker': ticker,
                        'source': 'Yahoo Finance',
                        'scraped_at': datetime.now()
                    })
            
            df = pd.DataFrame(news_items)
            
            # Save to cache
            if not df.empty:
                df.to_csv(cache_file, index=False)
                logger.info(f"Scraped {len(df)} Yahoo Finance news headlines for {ticker}")
            else:
                logger.warning(f"No news headlines found for {ticker} on Yahoo Finance")
                
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance news for {ticker}: {e}")
            
            # If error occurs, try to use cached data
            if os.path.exists(cache_file):
                logger.info(f"Using cached Yahoo Finance news for {ticker}")
                return pd.read_csv(cache_file)
                
            return pd.DataFrame()
    
    def scrape_multiple_tickers(self, tickers):
        """
        Scrape news for multiple tickers
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame with news for all tickers
        """
        all_news = []
        
        for ticker in tickers:
            news_df = self.scrape_yahoo_finance_news(ticker)
            if not news_df.empty:
                all_news.append(news_df)
                
        if all_news:
            return pd.concat(all_news, ignore_index=True)
        else:
            return pd.DataFrame()