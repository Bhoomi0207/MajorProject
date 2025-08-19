# Loading Historical Datasets for Stock Sentiment Analysis


import os
import time
import logging
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()  

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """
    Class to collect historical data from various free sources
    """
    def __init__(self, data_dir="data"):
        """
        Initialize with directories for storing data
        """
        # Create directories if they don't exist
        self.data_dir = data_dir
        self.hist_dir = os.path.join(data_dir, "historical")
        os.makedirs(self.hist_dir, exist_ok=True)
        os.makedirs(os.path.join(self.hist_dir, "prices"), exist_ok=True)
        os.makedirs(os.path.join(self.hist_dir, "reddit"), exist_ok=True)
        os.makedirs(os.path.join(self.hist_dir, "news"), exist_ok=True)
        
    def get_historical_prices(self, ticker, period="1y", interval="1d", save=True):
        """
        Get historical price data from Yahoo Finance
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        period : str
            Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval : str
            Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        pandas.DataFrame
            Historical price data
        """
        try:
            logger.info(f"Fetching historical price data for {ticker} ({period})")
            
            # Get data from Yahoo Finance
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No historical price data found for {ticker}")
                return None
                
            # Add ticker column
            hist['ticker'] = ticker
            
            # Reset index to make Date a column
            hist = hist.reset_index()
            
            # Format date column consistently
            if 'Date' in hist.columns:
                hist['Date'] = pd.to_datetime(hist['Date']).dt.date
            elif 'Datetime' in hist.columns:
                hist['Date'] = pd.to_datetime(hist['Datetime']).dt.date
                hist = hist.drop('Datetime', axis=1)
                
            logger.info(f"Retrieved {len(hist)} historical price records for {ticker}")
            
            if save:
                # Save to CSV
                filename = os.path.join(self.hist_dir, "prices", f"{ticker}_historical_prices.csv")
                hist.to_csv(filename, index=False)
                logger.info(f"Saved historical price data to {filename}")
                
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching historical price data for {ticker}: {e}")
            return None
            
    def get_historical_reddit(self, ticker, subreddit="wallstreetbets", days_back=180, limit=500, save=True):
        """
        Get historical Reddit posts using Pushshift API
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        subreddit : str
            Subreddit to search
        days_back : int
            Number of days to look back
        limit : int
            Maximum number of posts to retrieve
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        pandas.DataFrame
            Historical Reddit posts
        """
        try:
            logger.info(f"Fetching historical Reddit data for {ticker} from r/{subreddit} ({days_back} days back)")
            
            # Calculate start timestamp
            start_date = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            # Make request to Pushshift API
            url = f"https://api.pushshift.io/reddit/search/submission?q={ticker}&subreddit={subreddit}&after={start_date}&size={limit}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch Reddit data: Status code {response.status_code}")
                return None
                
            data = response.json()
            
            if 'data' not in data or len(data['data']) == 0:
                logger.warning(f"No historical Reddit posts found for {ticker} in r/{subreddit}")
                return None
                
            # Convert to DataFrame
            posts = pd.DataFrame(data['data'])
            
            # Select and rename important columns
            cols_to_keep = ['id', 'title', 'selftext', 'created_utc', 'score', 'num_comments', 'upvote_ratio']
            rename_map = {'created_utc': 'timestamp'}
            
            if not all(col in posts.columns for col in cols_to_keep):
                available_cols = [col for col in cols_to_keep if col in posts.columns]
                posts = posts[available_cols]
            else:
                posts = posts[cols_to_keep]
                
            # Rename columns
            for old_col, new_col in rename_map.items():
                if old_col in posts.columns:
                    posts = posts.rename(columns={old_col: new_col})
            
            # Add ticker and subreddit columns
            posts['ticker'] = ticker
            posts['subreddit'] = subreddit
            
            # Convert timestamp to datetime
            if 'timestamp' in posts.columns:
                posts['timestamp'] = pd.to_datetime(posts['timestamp'], unit='s')
                posts['date'] = posts['timestamp'].dt.date
            
            logger.info(f"Retrieved {len(posts)} historical Reddit posts for {ticker} from r/{subreddit}")
            
            if save:
                # Save to CSV
                filename = os.path.join(self.hist_dir, "reddit", f"{ticker}_{subreddit}_historical.csv")
                posts.to_csv(filename, index=False)
                logger.info(f"Saved historical Reddit data to {filename}")
                
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching historical Reddit data: {e}")
            return None
    
    def get_historical_news(self, ticker, days_back=30, save=True):
        """
        Get historical news data using free NewsAPI archive
        Note: Free NewsAPI only allows access to ~1 month of news history
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days_back : int
            Number of days to look back (max 30 for free API)
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        pandas.DataFrame
            Historical news articles
        """
        try:
            # Check for NewsAPI key in environment
            api_key = os.environ.get("NEWS_API_KEY")
            if not api_key:
                logger.warning("NEWS_API_KEY not found in environment variables")
                return None
                
            # Calculate date range (NewsAPI format YYYY-MM-DD)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=min(days_back, 30))  # Limit to 30 days for free tier
            
            logger.info(f"Fetching historical news for {ticker} from {start_date} to {end_date}")
            
            # Construct query (company name might yield better results than ticker)
            company_names = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google OR Alphabet',
                'AMZN': 'Amazon',
                'META': 'Meta OR Facebook',
                'JPM': 'JPMorgan',
                'V': 'Visa',
                # Add more mappings as needed
            }
            
            query = company_names.get(ticker, ticker)
            
            # Make request to NewsAPI
            url = f"https://newsapi.org/v2/everything?q={query}&from={start_date}&to={end_date}&language=en&sortBy=publishedAt&apiKey={api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch news data: Status code {response.status_code}")
                return None
                
            data = response.json()
            
            if data.get('status') != 'ok' or 'articles' not in data or len(data['articles']) == 0:
                logger.warning(f"No historical news found for {ticker}")
                return None
                
            # Convert to DataFrame
            articles = pd.DataFrame(data['articles'])
            
            # Add ticker column
            articles['ticker'] = ticker
            
            # Convert publishedAt to datetime
            if 'publishedAt' in articles.columns:
                articles['publishedAt'] = pd.to_datetime(articles['publishedAt'])
                articles['date'] = articles['publishedAt'].dt.date
            
            logger.info(f"Retrieved {len(articles)} historical news articles for {ticker}")
            
            if save:
                # Save to CSV
                filename = os.path.join(self.hist_dir, "news", f"{ticker}_historical_news.csv")
                articles.to_csv(filename, index=False)
                logger.info(f"Saved historical news data to {filename}")
                
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching historical news data: {e}")
            return None
            
    def fetch_all_historical(self, ticker, reddit_days=180, price_period="1y", news_days=30,
                           subreddits=None):
        """
        Fetch all historical data for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        reddit_days : int
            Days of Reddit history to fetch
        price_period : str
            Period for price history
        news_days : int
            Days of news history to fetch
        subreddits : list
            List of subreddits to search
            
        Returns:
        --------
        dict
            Dictionary containing all historical data
        """
        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing"]
            
        results = {
            "prices": None,
            "reddit": [],
            "news": None
        }
        
        # Fetch price data
        results["prices"] = self.get_historical_prices(ticker, period=price_period)
        
        # Fetch Reddit data from multiple subreddits
        for subreddit in subreddits:
            reddit_data = self.get_historical_reddit(ticker, subreddit=subreddit, days_back=reddit_days)
            if reddit_data is not None:
                results["reddit"].append(reddit_data)
                # Add a small delay to avoid overwhelming the API
                time.sleep(1)
                
        # Combine Reddit data if available
        if results["reddit"]:
            results["reddit"] = pd.concat(results["reddit"], ignore_index=True)
        else:
            results["reddit"] = None
            
        # Fetch news data
        results["news"] = self.get_historical_news(ticker, days_back=news_days)
        
        return results
    
    def create_merged_dataset(self, ticker, lookback_days=180):
        """
        Create a merged dataset aligning price, sentiment and news data by date
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        lookback_days : int
            Number of days to look back
            
        Returns:
        --------
        pandas.DataFrame
            Merged dataset with all available data
        """
        try:
            # First check if we have the data saved already
            price_file = os.path.join(self.hist_dir, "prices", f"{ticker}_historical_prices.csv")
            
            if not os.path.exists(price_file):
                logger.info(f"Historical price data for {ticker} not found. Fetching...")
                self.get_historical_prices(ticker, period="1y")
                
            if not os.path.exists(price_file):
                logger.error(f"Failed to fetch historical price data for {ticker}")
                return None
                
            # Load price data
            price_data = pd.read_csv(price_file)
            price_data['Date'] = pd.to_datetime(price_data['Date']).dt.date
            price_data = price_data.set_index('Date')
            
            # Create a date range dataframe as the base
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            date_range = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
            date_range.index = date_range.index.date
            
            # Merge price data with date range
            merged_data = date_range.join(price_data)
            
            # Load and process Reddit data if available
            reddit_files = [f for f in os.listdir(os.path.join(self.hist_dir, "reddit")) 
                          if f.startswith(f"{ticker}_") and f.endswith("_historical.csv")]
            
            if reddit_files:
                # Combine all Reddit files
                reddit_dfs = []
                for file in reddit_files:
                    df = pd.read_csv(os.path.join(self.hist_dir, "reddit", file))
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        reddit_dfs.append(df)
                
                if reddit_dfs:
                    combined_reddit = pd.concat(reddit_dfs, ignore_index=True)
                    
                    # Aggregate Reddit data by date
                    reddit_daily = combined_reddit.groupby('date').agg({
                        'score': 'mean',
                        'num_comments': 'sum',
                        'id': 'count'  # Count of posts
                    }).rename(columns={'id': 'post_count'})
                    
                    # Merge with main dataset
                    merged_data = merged_data.join(reddit_daily)
            
            # Load and process news data if available
            news_file = os.path.join(self.hist_dir, "news", f"{ticker}_historical_news.csv")
            if os.path.exists(news_file):
                news_data = pd.read_csv(news_file)
                if 'date' in news_data.columns:
                    news_data['date'] = pd.to_datetime(news_data['date']).dt.date
                    
                    # Aggregate news by date - just count for now
                    news_daily = news_data.groupby('date').size().to_frame('news_count')
                    
                    # Merge with main dataset
                    merged_data = merged_data.join(news_daily)
            
            # Forward fill missing price data (weekends/holidays)
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_columns:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].ffill()
            
            # Reset index to make Date a column again
            merged_data = merged_data.reset_index().rename(columns={'index': 'Date'})
            # Fill remaining NaN values with 0 for count columns
            count_columns = ['post_count', 'news_count', 'num_comments']
            for col in count_columns:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].fillna(0)
            
            logger.info(f"Created merged historical dataset for {ticker} with {len(merged_data)} days")
            
            # Save the merged dataset
            output_file = os.path.join(self.hist_dir, f"{ticker}_merged_historical.csv")
            merged_data.to_csv(output_file, index=False)
            logger.info(f"Saved merged historical dataset to {output_file}")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error creating merged dataset for {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def batch_collect_historical(self, tickers, lookback_days=180, include_news=True):
        """
        Collect historical data for multiple tickers
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        lookback_days : int
            Number of days to look back
        include_news : bool
            Whether to include news data (has API limitations)
            
        Returns:
        --------
        dict
            Dictionary mapping tickers to their merged datasets
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            logger.info(f"Processing historical data for {ticker} ({i+1}/{len(tickers)})...")
            
            try:
                # Fetch price data (always needed)
                self.get_historical_prices(ticker, period="1y")
                
                # Fetch Reddit data from multiple subreddits
                for subreddit in ["wallstreetbets", "stocks", "investing"]:
                    self.get_historical_reddit(ticker, subreddit=subreddit, days_back=lookback_days)
                    # Add a small delay between requests
                    time.sleep(2)
                
                # Fetch news data if requested
                if include_news:
                    self.get_historical_news(ticker, days_back=min(30, lookback_days))  # Limited to 30 days on free tier
                
                # Create the merged dataset
                merged_data = self.create_merged_dataset(ticker, lookback_days)
                results[ticker] = merged_data
                
                # Add a delay between tickers to avoid API rate limits
                if i < len(tickers) - 1:
                    delay = 5
                    logger.info(f"Waiting {delay} seconds before next ticker...")
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error processing historical data for {ticker}: {e}")
                results[ticker] = None
        
        return results


def load_historical_dataset(ticker, data_dir="data/historical", merge=True):
    """
    Load historical dataset for a specific ticker
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data_dir : str
        Directory where historical data is stored
    merge : bool
        Whether to return the merged dataset or individual components
        
    Returns:
    --------
    pandas.DataFrame or dict
        Merged dataset or dictionary of individual datasets
    """
    result = {}
    
    # Check for merged dataset first
    merged_file = os.path.join(data_dir, f"{ticker}_merged_historical.csv")
    if os.path.exists(merged_file) and merge:
        logger.info(f"Loading merged historical dataset for {ticker}")
        return pd.read_csv(merged_file)
    
    # Load price data
    price_file = os.path.join(data_dir, "prices", f"{ticker}_historical_prices.csv")
    if os.path.exists(price_file):
        result['prices'] = pd.read_csv(price_file)
        logger.info(f"Loaded historical price data for {ticker}: {len(result['prices'])} records")
    
    # Load Reddit data
    reddit_files = [f for f in os.listdir(os.path.join(data_dir, "reddit")) 
                  if f.startswith(f"{ticker}_") and f.endswith("_historical.csv")]
    
    if reddit_files:
        reddit_dfs = []
        for file in reddit_files:
            df = pd.read_csv(os.path.join(data_dir, "reddit", file))
            reddit_dfs.append(df)
        
        if reddit_dfs:
            result['reddit'] = pd.concat(reddit_dfs, ignore_index=True)
            logger.info(f"Loaded historical Reddit data for {ticker}: {len(result['reddit'])} posts")
    
    # Load news data
    news_file = os.path.join(data_dir, "news", f"{ticker}_historical_news.csv")
    if os.path.exists(news_file):
        result['news'] = pd.read_csv(news_file)
        logger.info(f"Loaded historical news data for {ticker}: {len(result['news'])} articles")
    
    if merge and 'prices' in result:
        # Create a simple merged dataset on the fly
        collector = HistoricalDataCollector(data_dir=os.path.dirname(data_dir))
        merged = collector.create_merged_dataset(ticker)
        return merged
    
    return result