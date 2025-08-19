import requests
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAPICollector:
    """
    Class to collect news articles from NewsAPI.org
    
    Sign up for a free API key at: https://newsapi.org/
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/"
        
        # Create data directory
        os.makedirs("data/news_api", exist_ok=True)
        
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
            'JPM': 'JPMorgan Chase',
            'V': 'Visa Inc',
            'BAC': 'Bank of America',
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer',
            'UNH': 'UnitedHealth Group',
            'XOM': 'Exxon Mobil',
            'CVX': 'Chevron',
            'CAT': 'Caterpillar',
            'BA': 'Boeing',
            'DIS': 'Disney',
            # Add more mappings as needed
        }
        
        # Use company name if available, otherwise use ticker
        search_term = company_names.get(ticker, ticker)
        
        # Construct endpoint
        endpoint = f"{self.base_url}everything"
        
        params = {
            'q': f'({search_term}) AND ({ticker})',
            'from': from_date,
            'to': to_date,
            'language': language,
            'sortBy': 'relevancy',
            'apiKey': self.api_key
        }
        
        try:
            logger.info(f"Fetching news articles for {ticker} from {from_date} to {to_date}...")
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "ok":
                logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            articles = data.get("articles", [])
            
            if not articles:
                logger.warning(f"No news articles found for {ticker}")
                return None
                
            records = []
            for article in articles:
                records.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "content": article.get("content"),
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
            
            logger.info(f"Successfully fetched {len(df)} news articles for {ticker}")
            
            if save:
                filename = f"data/news_api/{ticker}_news_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved news data to {filename}")
                
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news data: {e}")
            return None