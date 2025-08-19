import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class NewsAPICollector:
    """
    Class to collect news articles from NewsAPI.org for stock sentiment analysis
    """
    def __init__(self, api_key=None):
        # Load environment variables
        load_dotenv()
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")
        
        if not self.api_key:
            logger.error("No News API key provided. Please set NEWS_API_KEY environment variable.")
            raise ValueError("News API key is required")
            
        self.base_url = "https://newsapi.org/v2"
        
        # Create data directory
        os.makedirs("data/news", exist_ok=True)
        
    def fetch_stock_news(self, ticker, days_back=7, language="en", save=True):
        """
        Fetch news articles related to a stock ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days_back : int
            Number of days to look back for articles
        language : str
            Language of articles (e.g., 'en' for English)
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing news articles
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        # Company names mapping for better search results
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google OR Alphabet',
            'AMZN': 'Amazon',
            'META': 'Meta OR Facebook',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
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
            'IBM': 'IBM',
            'INTC': 'Intel',
            # Add more mappings as needed
        }
        
        # Use company name if available, otherwise use ticker
        search_term = company_names.get(ticker, ticker)
        
        # Construct endpoint for "everything" search
        endpoint = f"{self.base_url}/everything"
        
        # Define parameters for API request
        params = {
            'q': f'({search_term}) AND (stock OR shares OR market OR trading OR earnings OR investors)',
            'from': from_date,
            'to': to_date,
            'language': language,
            'sortBy': 'relevancy',
            'pageSize': 100,  # Maximum allowed by News API
            'apiKey': self.api_key
        }
        
        try:
            logger.info(f"Fetching news articles for {ticker} from {from_date} to {to_date}...")
            response = requests.get(endpoint, params=params)
            
            if response.status_code != 200:
                logger.error(f"API error: Status code {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            data = response.json()
            
            if data["status"] != "ok":
                logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            articles = data.get("articles", [])
            total_results = data.get("totalResults", 0)
            
            if not articles:
                logger.warning(f"No news articles found for {ticker}")
                return None
                
            logger.info(f"Found {total_results} articles, retrieved {len(articles)}")
            
            # Process articles into records
            records = []
            for article in articles:
                # Extract the main content without truncation if possible
                content = article.get("content", "")
                if content and "[+" in content:
                    content = content.split("[+")[0].strip()
                    
                records.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": content,
                    "source": article.get("source", {}).get("name"),
                    "author": article.get("author"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "ticker": ticker
                })
                
            df = pd.DataFrame(records)
            
            # Convert published_at to datetime
            if "published_at" in df.columns:
                df["published_at"] = pd.to_datetime(df["published_at"])
            
            logger.info(f"Successfully processed {len(df)} news articles for {ticker}")
            
            # Save to CSV and JSON if requested
            if save:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename_base = f"data/news/{ticker}_news_{timestamp}"
                
                # Save as CSV
                csv_filename = f"{filename_base}.csv"
                df.to_csv(csv_filename, index=False)
                logger.info(f"Saved news data to CSV: {csv_filename}")
                
                # Save raw JSON data
                json_filename = f"{filename_base}_raw.json"
                with open(json_filename, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved raw news data to JSON: {json_filename}")
                
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {e}")
            return None
    
    def analyze_sentiment_keywords(self, df):
        """
        Basic keyword-based sentiment analysis on news articles
        
        This is a simple implementation - a more sophisticated approach would
        use natural language processing models
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing news articles
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment scores added
        """
        if df is None or df.empty:
            return None
            
        # Define sentiment keywords
        positive_keywords = [
            'bullish', 'rally', 'growth', 'gain', 'positive', 'profit', 'surge',
            'jump', 'beat', 'exceed', 'rise', 'up', 'outperform', 'strong', 
            'opportunity', 'success', 'boost', 'soar', 'climb', 'recovery'
        ]
        
        negative_keywords = [
            'bearish', 'slump', 'decline', 'loss', 'negative', 'drop', 'fall',
            'sink', 'miss', 'plunge', 'down', 'underperform', 'weak', 'risk', 
            'fail', 'cut', 'layoffs', 'concern', 'struggle', 'crash'
        ]
        
        # Create copies to avoid modification warnings
        df_result = df.copy()
        
        # Function to count keyword occurrences
        def count_keywords(text, keyword_list):
            if not isinstance(text, str):
                return 0
            return sum(1 for keyword in keyword_list if keyword in text.lower())
        
        # Add sentiment columns based on title and content
        for idx, row in df_result.iterrows():
            # Combine title and content for analysis
            full_text = ' '.join(filter(None, [
                str(row.get('title', '')), 
                str(row.get('description', '')),
                str(row.get('content', ''))
            ])).lower()
            
            # Count positive and negative keywords
            positive_count = count_keywords(full_text, positive_keywords)
            negative_count = count_keywords(full_text, negative_keywords)
            
            # Calculate simple sentiment score
            if positive_count > negative_count:
                sentiment = 'positive'
                score = min(1.0, (positive_count - negative_count) / 10)
            elif negative_count > positive_count:
                sentiment = 'negative'
                score = min(1.0, (negative_count - positive_count) / 10) * -1
            else:
                sentiment = 'neutral'
                score = 0.0
            
            # Add to dataframe
            df_result.at[idx, 'positive_keywords'] = positive_count
            df_result.at[idx, 'negative_keywords'] = negative_count
            df_result.at[idx, 'sentiment'] = sentiment
            df_result.at[idx, 'sentiment_score'] = score
            
        return df_result
    
    def fetch_multiple_stocks(self, tickers, days_back=7, save=True):
        """
        Fetch news for multiple stock tickers
        
        Parameters:
        -----------
        tickers : list
            List of stock ticker symbols
        days_back : int
            Number of days to look back for articles
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        dict
            Dictionary of DataFrames containing news articles for each ticker
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            df = self.fetch_stock_news(ticker, days_back, save=save)
            
            if df is not None and not df.empty:
                # Add sentiment analysis
                df = self.analyze_sentiment_keywords(df)
                results[ticker] = df
                
                # Print summary
                if df is not None and not df.empty:
                    sentiment_counts = df['sentiment'].value_counts()
                    logger.info(f"Sentiment summary for {ticker}:")
                    for sentiment, count in sentiment_counts.items():
                        logger.info(f"  {sentiment}: {count}")
            
        return results


def main():
    """Main function to run the script from command line"""
    parser = argparse.ArgumentParser(description="Fetch news articles for stocks using News API")
    parser.add_argument("--tickers", "-t", type=str, required=True, 
                      help="Stock tickers (comma-separated, e.g., 'AAPL,MSFT,TSLA')")
    parser.add_argument("--days", "-d", type=int, default=7,
                      help="Number of days to look back for articles (default: 7)")
    parser.add_argument("--api-key", "-k", type=str,
                      help="News API key (optional, will use NEWS_API_KEY env var if not provided)")
    parser.add_argument("--save", "-s", action="store_true", default=True,
                      help="Save results to CSV and JSON (default: True)")
    
    args = parser.parse_args()
    
    # Split ticker string into list
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",")]
    
    try:
        # Initialize collector
        collector = NewsAPICollector(api_key=args.api_key)
        
        # Fetch news for all tickers
        results = collector.fetch_multiple_stocks(tickers, days_back=args.days, save=args.save)
        
        # Display summary
        logger.info("\nSummary:")
        for ticker, df in results.items():
            if df is not None:
                logger.info(f"{ticker}: {len(df)} articles collected")
            else:
                logger.info(f"{ticker}: No articles found")
        
    except Exception as e:
        logger.error(f"Error running news collection: {e}")
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