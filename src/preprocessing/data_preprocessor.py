import os
import pandas as pd
import numpy as np
import json
import logging
import re
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources (run once)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

class StockDataPreprocessor:
    """
    Class for preprocessing stock-related data from multiple sources
    """
    def __init__(self, data_dir="data"):
        """
        Initialize with directory containing data
        """
        self.data_dir = data_dir
        self.historical_dir = os.path.join(data_dir, "historical")
        self.prices_dir = os.path.join(self.historical_dir, "prices")
        self.news_dir = os.path.join(data_dir, "news")
        self.reddit_dir = os.path.join(data_dir, "reddit")
        self.stocks_dir = os.path.join(data_dir, "stocks")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
        
        # Initialize lemmatizer and stop words
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Stock-specific keywords for sentiment enhancement
        self.bullish_words = ['buy', 'bullish', 'upside', 'growth', 'positive', 'strong', 'outperform', 
                           'beat', 'upgrade', 'surged', 'rally', 'soar', 'gain', 'winner', 'optimistic']
        
        self.bearish_words = ['sell', 'bearish', 'downside', 'decline', 'negative', 'weak', 'underperform', 
                           'miss', 'downgrade', 'plunged', 'drop', 'sink', 'loss', 'loser', 'pessimistic']
    
    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True):
        """
        Basic text preprocessing
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Rejoin tokens
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def enhanced_sentiment_analysis(self, text, ticker=None):
        """
        Enhanced sentiment analysis with stock-specific adjustments
        """
        if not isinstance(text, str) or text.strip() == "":
            return {
                'compound': 0,
                'pos': 0,
                'neu': 1,
                'neg': 0,
                'sentiment_label': 'neutral'
            }
        
        # Get base sentiment scores
        sentiment = self.sid.polarity_scores(text)
        
        # Count bullish and bearish words
        text_lower = text.lower()
        bullish_count = sum(1 for word in self.bullish_words if word in text_lower)
        bearish_count = sum(1 for word in self.bearish_words if word in text_lower)
        
        # Boost sentiment based on stock-specific terms
        sentiment_boost = (bullish_count - bearish_count) * 0.05
        
        # Apply ticker-specific boost if ticker is mentioned
        ticker_boost = 0
        if ticker and ticker.lower() in text_lower:
            ticker_boost = 0.1 if sentiment['compound'] > 0 else -0.1 if sentiment['compound'] < 0 else 0
        
        # Adjust compound score (ensuring it stays within -1 to 1)
        adjusted_compound = max(min(sentiment['compound'] + sentiment_boost + ticker_boost, 1), -1)
        
        # Update sentiment dictionary
        sentiment['compound'] = adjusted_compound
        
        # Add sentiment label
        if sentiment['compound'] >= 0.05:
            sentiment['sentiment_label'] = 'positive'
        elif sentiment['compound'] <= -0.05:
            sentiment['sentiment_label'] = 'negative'
        else:
            sentiment['sentiment_label'] = 'neutral'
            
        return sentiment
    
    def load_and_preprocess_historical_price(self, ticker):
        """
        Load and preprocess historical price data for a ticker
        """
        # Try to load merged historical first
        merged_file = os.path.join(self.historical_dir, f"{ticker}_merged_historical.csv")
        price_file = os.path.join(self.prices_dir, f"{ticker}_historical_prices.csv")
        
        if os.path.exists(merged_file):
            logger.info(f"Loading merged historical data for {ticker} from {merged_file}")
            try:
                df = pd.read_csv(merged_file)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                return df
            except Exception as e:
                logger.error(f"Error loading merged historical file: {e}")
        
        elif os.path.exists(price_file):
            logger.info(f"Loading price data for {ticker} from {price_file}")
            try:
                df = pd.read_csv(price_file)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                return df
            except Exception as e:
                logger.error(f"Error loading price file: {e}")
        
        else:
            logger.warning(f"No historical price data found for {ticker}")
            return None
    
    def load_and_preprocess_reddit(self, ticker):
        """
        Load and preprocess Reddit data for a ticker
        """
        # Find all Reddit files for this ticker
        reddit_files = [f for f in os.listdir(self.reddit_dir) 
                        if f.startswith(f"{ticker}_") and f.endswith(".csv")]
        
        if not reddit_files:
            logger.warning(f"No Reddit data found for {ticker}")
            return None
        
        all_reddit_data = []
        
        for file in reddit_files:
            try:
                file_path = os.path.join(self.reddit_dir, file)
                logger.info(f"Loading Reddit data from {file_path}")
                
                df = pd.read_csv(file_path)
                all_reddit_data.append(df)
            except Exception as e:
                logger.error(f"Error loading Reddit file {file}: {e}")
        
        if not all_reddit_data:
            return None
        
        # Combine all Reddit data
        combined_df = pd.concat(all_reddit_data, ignore_index=True)
        
        # Process text columns
        if 'text' in combined_df.columns:
            combined_df['processed_text'] = combined_df['text'].apply(self.preprocess_text)
        elif 'selftext' in combined_df.columns:
            combined_df['processed_text'] = combined_df['selftext'].apply(self.preprocess_text)
        
        if 'title' in combined_df.columns:
            combined_df['processed_title'] = combined_df['title'].apply(self.preprocess_text)
        
        # Create combined text field
        if 'processed_text' in combined_df.columns and 'processed_title' in combined_df.columns:
            combined_df['combined_text'] = combined_df['processed_title'] + ' ' + combined_df['processed_text']
        elif 'processed_title' in combined_df.columns:
            combined_df['combined_text'] = combined_df['processed_title']
        elif 'processed_text' in combined_df.columns:
            combined_df['combined_text'] = combined_df['processed_text']
        
        # Apply sentiment analysis
        if 'combined_text' in combined_df.columns:
            logger.info(f"Applying sentiment analysis to {len(combined_df)} Reddit posts")
            
            # Apply enhanced sentiment analysis
            sentiments = combined_df['combined_text'].apply(
                lambda x: self.enhanced_sentiment_analysis(x, ticker)
            )
            
            # Extract sentiment scores
            combined_df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
            combined_df['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
            combined_df['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
            combined_df['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])
            combined_df['sentiment_label'] = sentiments.apply(lambda x: x['sentiment_label'])
        
        # Ensure created_utc is datetime and create 'date' column
        if 'created_utc' in combined_df.columns:
            combined_df['created_utc'] = pd.to_datetime(combined_df['created_utc'], unit='s', errors='coerce')
            combined_df = combined_df.dropna(subset=['created_utc'])
            combined_df['date'] = combined_df['created_utc'].dt.date
        else:
            logger.error(f"No 'created_utc' column found in Reddit data for {ticker}. Cannot create 'date' column.")
            return None
        
        return combined_df
    
    def load_and_preprocess_news(self, ticker):
        """
        Load and preprocess news data for a ticker
        """
        # Find news files
        news_files = [f for f in os.listdir(self.news_dir) if f.endswith(".json")]
        
        if not news_files:
            logger.warning(f"No news data found in {self.news_dir}")
            return None
        
        all_news_data = []
        
        for file in news_files:
            try:
                file_path = os.path.join(self.news_dir, file)
                logger.info(f"Loading news data from {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    news_data = json.load(f)
                
                if 'articles' in news_data:
                    # Convert to DataFrame
                    articles_df = pd.DataFrame(news_data['articles'])
                    
                    # Filter for articles mentioning this ticker
                    ticker_variant = ticker.replace('.', '')
                    filtered_articles = articles_df[
                        articles_df['title'].str.contains(ticker, case=False, na=False) | 
                        articles_df['title'].str.contains(ticker_variant, case=False, na=False) |
                        articles_df['description'].str.contains(ticker, case=False, na=False) |
                        articles_df['description'].str.contains(ticker_variant, case=False, na=False)
                    ].copy()
                    
                    if not filtered_articles.empty:
                        all_news_data.append(filtered_articles)
            except Exception as e:
                logger.error(f"Error loading news file {file}: {e}")
        
        if not all_news_data:
            logger.warning(f"No relevant news found for {ticker}")
            return None
        
        # Combine all news data
        combined_df = pd.concat(all_news_data, ignore_index=True)
        
        # Process text columns
        if 'title' in combined_df.columns:
            combined_df['processed_title'] = combined_df['title'].apply(self.preprocess_text)
        
        if 'description' in combined_df.columns:
            combined_df['processed_description'] = combined_df['description'].apply(self.preprocess_text)
        
        if 'content' in combined_df.columns:
            combined_df['processed_content'] = combined_df['content'].apply(self.preprocess_text)
        
        # Create combined text field
        combined_df['combined_text'] = ''
        
        if 'processed_title' in combined_df.columns:
            combined_df['combined_text'] += combined_df['processed_title'] + ' '
        
        if 'processed_description' in combined_df.columns:
            combined_df['combined_text'] += combined_df['processed_description'] + ' '
        
        if 'processed_content' in combined_df.columns:
            combined_df['combined_text'] += combined_df['processed_content']
        
        # Apply sentiment analysis
        if 'combined_text' in combined_df.columns:
            logger.info(f"Applying sentiment analysis to {len(combined_df)} news articles")
            
            # Apply enhanced sentiment analysis
            sentiments = combined_df['combined_text'].apply(
                lambda x: self.enhanced_sentiment_analysis(x, ticker)
            )
            
            # Extract sentiment scores
            combined_df['sentiment_compound'] = sentiments.apply(lambda x: x['compound'])
            combined_df['sentiment_positive'] = sentiments.apply(lambda x: x['pos'])
            combined_df['sentiment_neutral'] = sentiments.apply(lambda x: x['neu'])
            combined_df['sentiment_negative'] = sentiments.apply(lambda x: x['neg'])
            combined_df['sentiment_label'] = sentiments.apply(lambda x: x['sentiment_label'])
        
        # Ensure publishedAt is datetime and create 'date' column
        if 'publishedAt' in combined_df.columns:
            combined_df['publishedAt'] = pd.to_datetime(combined_df['publishedAt'], errors='coerce')
            combined_df = combined_df.dropna(subset=['publishedAt'])
            combined_df['date'] = combined_df['publishedAt'].dt.date
        else:
            logger.error(f"No 'publishedAt' column found in news data for {ticker}. Cannot create 'date' column.")
            return None
        
        # Add ticker column
        combined_df['ticker'] = ticker
        
        return combined_df
    
    def calculate_daily_sentiment(self, df, date_column, sentiment_column):
        """
        Calculate daily sentiment aggregates
        """
        if df is None or df.empty or date_column not in df.columns or sentiment_column not in df.columns:
            logger.error(f"DataFrame is empty or missing required columns: {date_column}, {sentiment_column}")
            return None
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        
        # Check if 'date' column exists after conversion
        if date_column not in df.columns:
            logger.error(f"Column '{date_column}' does not exist in the DataFrame after conversion.")
            return None
        
        # Group by date and calculate sentiment metrics
        daily_sentiment = df.groupby(df[date_column].dt.date).agg({
            sentiment_column: ['mean', 'median', 'count', 'std'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        })
        
        # Flatten multi-level columns
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
        
        # Reset index to make date a column
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment.rename(columns={
            'index': 'date',
            f'{sentiment_column}_count': 'message_count'
        }, inplace=True)
        
        return daily_sentiment
    
    def load_and_preprocess_stocks_data(self, ticker):
        """
        Load and preprocess stock price data with technical indicators
        """
        # Find stock data file
        stock_file = os.path.join(self.stocks_dir, f"{ticker}_data.csv")
        
        if not os.path.exists(stock_file):
            logger.warning(f"No stock data found for {ticker}")
            return None
        
        try:
            logger.info(f"Loading stock data from {stock_file}")
            df = pd.read_csv(stock_file)
            
            # Ensure Date is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            return df
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            return None
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators to price data if they don't exist already
        """
        if df is None or df.empty or 'Close' not in df.columns:
            return df
        
        # Check if indicators already exist
        existing_indicators = set(df.columns)
        
        # Moving Averages
        if 'SMA_5' not in existing_indicators:
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
        
        if 'SMA_20' not in existing_indicators:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        if 'EMA_12' not in existing_indicators:
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        
        if 'EMA_26' not in existing_indicators:
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        if 'MACD' not in existing_indicators:
            df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        if 'MACD_Signal' not in existing_indicators:
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        if 'MACD_Hist' not in existing_indicators:
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        if 'RSI' not in existing_indicators:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'BB_Middle' not in existing_indicators:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_StdDev'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_StdDev']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_StdDev']
        
        # Price momentum
        if 'Momentum_5d' not in existing_indicators:
            df['Momentum_5d'] = df['Close'].pct_change(periods=5)
        
        if 'Momentum_10d' not in existing_indicators:
            df['Momentum_10d'] = df['Close'].pct_change(periods=10)
        
        # Volatility (standard deviation of returns)
        if 'Volatility_10d' not in existing_indicators:
            df['Volatility_10d'] = df['Close'].pct_change().rolling(window=10).std()
        
        return df
    
    def merge_all_data(self, ticker):
        """
        Merge all data sources for a ticker
        """
        logger.info(f"Merging all data for {ticker}")
        
        # Load historical price data (base dataset)
        price_data = self.load_and_preprocess_historical_price(ticker)
        
        if price_data is None:
            logger.error(f"No price data found for {ticker}. Cannot proceed with merging.")
            return None
        
        # Ensure 'Date' column is datetime and clean it
        if 'Date' in price_data.columns:
            price_data['Date'] = pd.to_datetime(price_data['Date'], errors='coerce')
            price_data = price_data.dropna(subset=['Date'])
        else:
            logger.error(f"No 'Date' column found in historical price data for {ticker}")
            return None
        
        # Add technical indicators
        price_data = self.add_technical_indicators(price_data)
        
        # Load and preprocess Reddit data
        reddit_data = self.load_and_preprocess_reddit(ticker)
        
        # Calculate daily sentiment from Reddit data
        if reddit_data is not None:
            if 'date' not in reddit_data.columns or 'sentiment_compound' not in reddit_data.columns:
                logger.error(f"Reddit data for {ticker} is missing required columns. Skipping Reddit sentiment.")
            else:
                reddit_sentiment = self.calculate_daily_sentiment(reddit_data, 'date', 'sentiment_compound')
                if reddit_sentiment is not None:
                    price_data = pd.merge(price_data, reddit_sentiment, how='left', left_on='Date', right_on='date')
                    price_data.drop(columns=['date'], inplace=True)
        
        # Load and preprocess news data
        news_data = self.load_and_preprocess_news(ticker)
        
        # Calculate daily sentiment from news data
        if news_data is not None:
            if 'date' not in news_data.columns or 'sentiment_compound' not in news_data.columns:
                logger.error(f"News data for {ticker} is missing required columns. Skipping news sentiment.")
            else:
                news_sentiment = self.calculate_daily_sentiment(news_data, 'date', 'sentiment_compound')
                if news_sentiment is not None:
                    price_data = pd.merge(price_data, news_sentiment, how='left', left_on='Date', right_on='date')
                    price_data.drop(columns=['date'], inplace=True)
        
        # Save the merged data
        output_file = os.path.join(self.processed_dir, f"{ticker}_processed.csv")
        price_data.to_csv(output_file, index=False)
        logger.info(f"Merged data saved to {output_file}")
        
        return price_data
    
    def process_multiple_tickers(self, tickers):
        """
        Process multiple tickers
        """
        results = {}
        for ticker in tickers:
            try:
                logger.info(f"Processing {ticker}...")
                results[ticker] = self.merge_all_data(ticker)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                results[ticker] = None
        return results