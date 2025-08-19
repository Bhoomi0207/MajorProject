import tweepy
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import praw
import schedule
import logging
import json
import re
from typing import List, Dict, Any, Union
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_sentiment")

# Configuration - store in a separate config.py file in production
class Config:
    # Twitter API credentials
    TWITTER_BEARER_TOKEN = "YOUR_BEARER_TOKEN"
    TWITTER_API_KEY = "YOUR_API_KEY"
    TWITTER_API_SECRET = "YOUR_API_SECRET"
    TWITTER_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"
    TWITTER_ACCESS_SECRET = "YOUR_ACCESS_SECRET"
    
    # StockTwits credentials
    STOCKTWITS_API_KEY = "YOUR_STOCKTWITS_API_KEY"
    
    # Reddit credentials
    REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
    REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
    REDDIT_USER_AGENT = "script:stock_sentiment_analysis:v1.0 (by u/YOUR_USERNAME)"
    
    # List of stock tickers to track
    # You can expand this list or load from a file
    STOCK_TICKERS = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 
        'V', 'PG', 'JNJ', 'UNH', 'HD', 'MA', 'BAC', 'XOM', 'DIS', 'NFLX'
    ]
    
    # Mapping tickers to sectors - useful for sector analysis later
    # This is a simplified version, you can expand it
    SECTOR_MAPPING = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Consumer Cyclical',
        'GOOGL': 'Communication Services', 'META': 'Communication Services',
        'TSLA': 'Consumer Cyclical', 'NVDA': 'Technology', 'JPM': 'Financial Services',
        'V': 'Financial Services', 'PG': 'Consumer Defensive', 'JNJ': 'Healthcare',
        'UNH': 'Healthcare', 'HD': 'Consumer Cyclical', 'MA': 'Financial Services',
        'BAC': 'Financial Services', 'XOM': 'Energy', 'DIS': 'Communication Services',
        'NFLX': 'Communication Services'
    }
    
    # Data collection settings
    MAX_TWEETS_PER_QUERY = 10  # Twitter free API limit
    COLLECTION_INTERVAL_HOURS = 4  # Collect every 4 hours
    RATE_LIMIT_SLEEP = 2  # Seconds to sleep between API calls
    DATA_DIR = "collected_data"


class TwitterCollector:
    """Class to handle Twitter data collection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = tweepy.Client(
            bearer_token=config.TWITTER_BEARER_TOKEN,
            consumer_key=config.TWITTER_API_KEY,
            consumer_secret=config.TWITTER_API_SECRET,
            access_token=config.TWITTER_ACCESS_TOKEN,
            access_token_secret=config.TWITTER_ACCESS_SECRET
        )
    
    def collect_tweets_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Collect tweets for a specific ticker"""
        query = f"${ticker} lang:en -is:retweet"
        
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=self.config.MAX_TWEETS_PER_QUERY,
                tweet_fields=['created_at', 'public_metrics', 'author_id']
            )
            
            if not tweets.data:
                logger.info(f"No tweets found for {ticker}")
                return pd.DataFrame()
                
            tweet_data = []
            for tweet in tweets.data:
                tweet_data.append({
                    'id': tweet.id,
                    'ticker': ticker,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'author_id': tweet.author_id,
                    'source': 'twitter'
                })
                
            return pd.DataFrame(tweet_data)
        
        except Exception as e:
            logger.error(f"Error collecting tweets for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_all_tickers(self) -> pd.DataFrame:
        """Collect tweets for all tickers"""
        all_tweets = pd.DataFrame()
        
        for ticker in self.config.STOCK_TICKERS:
            logger.info(f"Collecting tweets for ${ticker}")
            ticker_tweets = self.collect_tweets_for_ticker(ticker)
            all_tweets = pd.concat([all_tweets, ticker_tweets])
            
            # Sleep to respect rate limits
            time.sleep(self.config.RATE_LIMIT_SLEEP)
        
        return all_tweets


class StockTwitsCollector:
    """Class to handle StockTwits data collection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "https://api.stocktwits.com/api/2"
    
    def collect_stocktwits_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Collect StockTwits posts for a specific ticker"""
        endpoint = f"{self.base_url}/streams/symbol/{ticker}.json"
        
        try:
            response = requests.get(endpoint)
            if response.status_code != 200:
                logger.error(f"Error fetching StockTwits for {ticker}: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'messages' not in data:
                logger.info(f"No StockTwits messages found for {ticker}")
                return pd.DataFrame()
            
            messages_data = []
            for message in data['messages']:
                messages_data.append({
                    'id': message['id'],
                    'ticker': ticker,
                    'text': message['body'],
                    'created_at': message['created_at'],
                    'like_count': message.get('likes', {}).get('total', 0),
                    'sentiment': message.get('entities', {}).get('sentiment', {}).get('basic', ''),
                    'user_followers': message.get('user', {}).get('followers', 0),
                    'source': 'stocktwits'
                })
            
            return pd.DataFrame(messages_data)
        
        except Exception as e:
            logger.error(f"Error collecting StockTwits for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_all_tickers(self) -> pd.DataFrame:
        """Collect StockTwits posts for all tickers"""
        all_messages = pd.DataFrame()
        
        for ticker in self.config.STOCK_TICKERS:
            logger.info(f"Collecting StockTwits for ${ticker}")
            ticker_messages = self.collect_stocktwits_for_ticker(ticker)
            all_messages = pd.concat([all_messages, ticker_messages])
            
            # Sleep to respect rate limits
            time.sleep(self.config.RATE_LIMIT_SLEEP)
        
        return all_messages


class YahooFinanceCollector:
    """Class to handle Yahoo Finance data collection"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def collect_stock_data(self, ticker: str) -> pd.DataFrame:
        """Collect stock market data for a ticker"""
        try:
            # Get stock data for the past 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if stock_data.empty:
                logger.info(f"No Yahoo Finance data found for {ticker}")
                return pd.DataFrame()
            
            # Reset index to make Date a column and add ticker
            stock_data = stock_data.reset_index()
            stock_data['ticker'] = ticker
            stock_data['source'] = 'yahoo_finance'
            
            return stock_data
        
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance data for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_news_headlines(self, ticker: str) -> pd.DataFrame:
        """Scrape Yahoo Finance news headlines for a ticker"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Error fetching Yahoo Finance news for {ticker}: {response.status_code}")
                return pd.DataFrame()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_elements = soup.find_all('h3', class_='Mb(5px)')
            
            news_data = []
            for element in news_elements:
                if element.a:
                    news_data.append({
                        'ticker': ticker,
                        'headline': element.a.text,
                        'created_at': datetime.now(),
                        'source': 'yahoo_finance_news'
                    })
            
            return pd.DataFrame(news_data)
        
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance news for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_all_tickers(self) -> Dict[str, pd.DataFrame]:
        """Collect Yahoo Finance data for all tickers"""
        stock_data = pd.DataFrame()
        news_data = pd.DataFrame()
        
        for ticker in self.config.STOCK_TICKERS:
            logger.info(f"Collecting Yahoo Finance data for {ticker}")
            
            # Collect stock price data
            ticker_stock_data = self.collect_stock_data(ticker)
            stock_data = pd.concat([stock_data, ticker_stock_data])
            
            # Collect news headlines
            ticker_news_data = self.collect_news_headlines(ticker)
            news_data = pd.concat([news_data, ticker_news_data])
            
            # Sleep to respect web scraping best practices
            time.sleep(self.config.RATE_LIMIT_SLEEP)
        
        return {'stock_data': stock_data, 'news_data': news_data}


class RedditCollector:
    """Class to handle Reddit data collection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT
        )
        self.subreddits = ['wallstreetbets', 'investing', 'stocks']
    
    def collect_reddit_posts(self, ticker: str) -> pd.DataFrame:
        """Collect Reddit posts mentioning a ticker"""
        all_posts = pd.DataFrame()
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for ticker in submissions
                search_query = f"{ticker} OR ${ticker}"
                submissions = subreddit.search(search_query, limit=5, sort='new')
                
                posts_data = []
                for submission in submissions:
                    posts_data.append({
                        'id': submission.id,
                        'ticker': ticker,
                        'title': submission.title,
                        'text': submission.selftext,
                        'created_at': datetime.fromtimestamp(submission.created_utc),
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'subreddit': subreddit_name,
                        'source': 'reddit'
                    })
                
                subreddit_posts = pd.DataFrame(posts_data)
                all_posts = pd.concat([all_posts, subreddit_posts])
                
            except Exception as e:
                logger.error(f"Error collecting Reddit data for {ticker} in r/{subreddit_name}: {e}")
        
        return all_posts
    
    def collect_all_tickers(self) -> pd.DataFrame:
        """Collect Reddit posts for all tickers"""
        all_posts = pd.DataFrame()
        
        for ticker in self.config.STOCK_TICKERS:
            logger.info(f"Collecting Reddit posts for {ticker}")
            ticker_posts = self.collect_reddit_posts(ticker)
            all_posts = pd.concat([all_posts, ticker_posts])
            
            # Sleep to respect rate limits
            time.sleep(self.config.RATE_LIMIT_SLEEP)
        
        return all_posts


class DataCollectionManager:
    """Main class to manage all data collection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.twitter_collector = TwitterCollector(config)
        self.stocktwits_collector = StockTwitsCollector(config)
        self.yahoo_collector = YahooFinanceCollector(config)
        self.reddit_collector = RedditCollector(config)
        
        # Create data directory if it doesn't exist
        os.makedirs(config.DATA_DIR, exist_ok=True)
    
    def collect_all_data(self) -> None:
        """Collect data from all sources"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting data collection cycle at {timestamp}")
        
        # Use ThreadPoolExecutor to collect data in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit collection tasks
            twitter_future = executor.submit(self.twitter_collector.collect_all_tickers)
            stocktwits_future = executor.submit(self.stocktwits_collector.collect_all_tickers)
            yahoo_future = executor.submit(self.yahoo_collector.collect_all_tickers)
            reddit_future = executor.submit(self.reddit_collector.collect_all_tickers)
            
            # Get results
            twitter_data = twitter_future.result()
            stocktwits_data = stocktwits_future.result()
            yahoo_data = yahoo_future.result()
            reddit_data = reddit_future.result()
        
        # Save data to CSV files
        if not twitter_data.empty:
            twitter_file = os.path.join(self.config.DATA_DIR, f"twitter_{timestamp}.csv")
            twitter_data.to_csv(twitter_file, index=False)
            logger.info(f"Saved {len(twitter)}")

        twitter_data.to_csv(twitter_file, index=False)
        logger.info(f"Saved {len(twitter_data)} Twitter posts to {twitter_file}")
        
        if not stocktwits_data.empty:
            stocktwits_file = os.path.join(self.config.DATA_DIR, f"stocktwits_{timestamp}.csv")
            stocktwits_data.to_csv(stocktwits_file, index=False)
            logger.info(f"Saved {len(stocktwits_data)} StockTwits messages to {stocktwits_file}")
        
        if 'stock_data' in yahoo_data and not yahoo_data['stock_data'].empty:
            yahoo_stock_file = os.path.join(self.config.DATA_DIR, f"yahoo_stock_{timestamp}.csv")
            yahoo_data['stock_data'].to_csv(yahoo_stock_file, index=False)
            logger.info(f"Saved {len(yahoo_data['stock_data'])} Yahoo Finance stock data points to {yahoo_stock_file}")
        
        if 'news_data' in yahoo_data and not yahoo_data['news_data'].empty:
            yahoo_news_file = os.path.join(self.config.DATA_DIR, f"yahoo_news_{timestamp}.csv")
            yahoo_data['news_data'].to_csv(yahoo_news_file, index=False)
            logger.info(f"Saved {len(yahoo_data['news_data'])} Yahoo Finance news headlines to {yahoo_news_file}")
        
        if not reddit_data.empty:
            reddit_file = os.path.join(self.config.DATA_DIR, f"reddit_{timestamp}.csv")
            reddit_data.to_csv(reddit_file, index=False)
            logger.info(f"Saved {len(reddit_data)} Reddit posts to {reddit_file}")
        
        # Create a metadata file with collection statistics
        metadata = {
            'timestamp': timestamp,
            'twitter_count': len(twitter_data),
            'stocktwits_count': len(stocktwits_data),
            'yahoo_stock_count': len(yahoo_data.get('stock_data', pd.DataFrame())),
            'yahoo_news_count': len(yahoo_data.get('news_data', pd.DataFrame())),
            'reddit_count': len(reddit_data),
            'tickers_collected': self.config.STOCK_TICKERS,
            'collection_duration': str(datetime.now() - datetime.strptime(timestamp, "%Y%m%d_%H%M%S"))
        }
        
        metadata_file = os.path.join(self.config.DATA_DIR, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Data collection cycle completed. Metadata saved to {metadata_file}")
    
    def merge_historical_data(self, days: int = 7) -> Dict[str, pd.DataFrame]:
        """Merge data collected over the past several days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        merged_data = {
            'twitter': pd.DataFrame(),
            'stocktwits': pd.DataFrame(),
            'yahoo_stock': pd.DataFrame(),
            'yahoo_news': pd.DataFrame(),
            'reddit': pd.DataFrame()
        }
        
        # Get all CSV files in the data directory
        files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('.csv')]
        
        for file in files:
            file_path = os.path.join(self.config.DATA_DIR, file)
            # Extract timestamp from filename
            try:
                timestamp_str = re.search(r'_(\d{8}_\d{6})\.csv', file).group(1)
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Only include files from the last 'days' days
                if file_date >= cutoff_date:
                    df = pd.read_csv(file_path)
                    
                    if file.startswith('twitter'):
                        merged_data['twitter'] = pd.concat([merged_data['twitter'], df])
                    elif file.startswith('stocktwits'):
                        merged_data['stocktwits'] = pd.concat([merged_data['stocktwits'], df])
                    elif file.startswith('yahoo_stock'):
                        merged_data['yahoo_stock'] = pd.concat([merged_data['yahoo_stock'], df])
                    elif file.startswith('yahoo_news'):
                        merged_data['yahoo_news'] = pd.concat([merged_data['yahoo_news'], df])
                    elif file.startswith('reddit'):
                        merged_data['reddit'] = pd.concat([merged_data['reddit'], df])
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
        
        # Remove duplicates
        for key in merged_data:
            if not merged_data[key].empty:
                merged_data[key] = merged_data[key].drop_duplicates()
                
        # Create merged files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for key, df in merged_data.items():
            if not df.empty:
                merged_file = os.path.join(self.config.DATA_DIR, f"merged_{key}_{timestamp}.csv")
                df.to_csv(merged_file, index=False)
                logger.info(f"Saved merged {key} data with {len(df)} entries to {merged_file}")
        
        return merged_data
    
    def schedule_collection(self) -> None:
        """Schedule regular data collection"""
        # Schedule collection at regular intervals
        schedule.every(self.config.COLLECTION_INTERVAL_HOURS).hours.do(self.collect_all_data)
        
        # Schedule weekly data merging
        schedule.every().sunday.at("23:00").do(self.merge_historical_data)
        
        logger.info(f"Scheduled data collection every {self.config.COLLECTION_INTERVAL_HOURS} hours")
        logger.info("Scheduled weekly data merging every Sunday at 23:00")
        
        # Run initial collection
        self.collect_all_data()
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)


class DataRotator:
    """Class to rotate through different stock groups to maximize API usage"""
    
    def __init__(self, config: Config):
        self.config = config
        self.all_tickers = config.STOCK_TICKERS
        
        # Group tickers by sector
        self.sector_groups = {}
        for ticker, sector in config.SECTOR_MAPPING.items():
            if sector not in self.sector_groups:
                self.sector_groups[sector] = []
            self.sector_groups[sector].append(ticker)
        
        # Create rotation schedule
        self.rotation_schedule = self._create_rotation_schedule()
        self.current_rotation_index = 0
    
    def _create_rotation_schedule(self) -> List[List[str]]:
        """Create a rotation schedule for tickers"""
        schedule = []
        
        # Add sector-based groups
        for sector, tickers in self.sector_groups.items():
            if tickers:
                schedule.append(tickers)
        
        # If any tickers are not in a sector, add them in small groups
        ungrouped = [t for t in self.all_tickers if t not in [item for sublist in schedule for item in sublist]]
        if ungrouped:
            # Split into groups of 5 or fewer
            for i in range(0, len(ungrouped), 5):
                schedule.append(ungrouped[i:i+5])
        
        return schedule
    
    def get_next_ticker_group(self) -> List[str]:
        """Get the next group of tickers in the rotation"""
        if not self.rotation_schedule:
            return self.all_tickers[:5]  # Fallback to first 5 tickers
        
        ticker_group = self.rotation_schedule[self.current_rotation_index]
        self.current_rotation_index = (self.current_rotation_index + 1) % len(self.rotation_schedule)
        
        return ticker_group


class DataAugmenter:
    """Class to augment collected data to increase dataset size"""
    
    def __init__(self):
        pass
    
    def augment_text(self, text: str) -> List[str]:
        """Simple text augmentation techniques"""
        augmented_texts = []
        
        # Original text
        augmented_texts.append(text)
        
        # Remove some random words (simulate Twitter shorthand)
        words = text.split()
        if len(words) > 4:
            import random
            remove_indices = random.sample(range(len(words)), k=min(2, len(words)//4))
            augmented_text = ' '.join([w for i, w in enumerate(words) if i not in remove_indices])
            augmented_texts.append(augmented_text)
        
        # Replace some words with synonyms
        # This is a simplified version - in a real application, you'd use a proper synonym library
        simple_synonyms = {
            'good': ['great', 'excellent', 'positive'],
            'bad': ['poor', 'terrible', 'negative'],
            'buy': ['purchase', 'acquire', 'get'],
            'sell': ['dump', 'offload', 'exit'],
            'stock': ['share', 'equity', 'position'],
            'market': ['exchange', 'trading', 'stocks'],
            'up': ['higher', 'rise', 'climb'],
            'down': ['lower', 'fall', 'drop']
        }
        
        for word, synonyms in simple_synonyms.items():
            if f' {word} ' in f' {text} ':
                for synonym in synonyms:
                    augmented_text = text.replace(f' {word} ', f' {synonym} ')
                    augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def augment_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Augment a dataframe with text data"""
        if text_column not in df.columns or df.empty:
            return df
        
        augmented_rows = []
        
        for _, row in df.iterrows():
            original_text = row[text_column]
            augmented_texts = self.augment_text(original_text)
            
            # Skip the first one as it's the original
            for i, aug_text in enumerate(augmented_texts[1:], 1):
                new_row = row.copy()
                new_row[text_column] = aug_text
                new_row['id'] = f"{new_row['id']}_aug{i}"
                new_row['augmented'] = True
                augmented_rows.append(new_row)
        
        # Add augmentation flag to original data
        df['augmented'] = False
        
        # Combine original and augmented data
        augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)])
        
        return augmented_df


if __name__ == "__main__":
    # Create configuration
    config = Config()
    
    # Create data collection manager
    manager = DataCollectionManager(config)
    
    # Run scheduled collection
    try:
        manager.schedule_collection()
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user")