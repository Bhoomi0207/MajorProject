import requests
import pandas as pd
import logging
from datetime import datetime
import os
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StocktwitsDataCollector:
    """
    Class to collect messages from StockTwits API
    with updated access method to handle restrictions
    """
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"
        # Create data directory if it doesn't exist
        os.makedirs("data/stocktwits", exist_ok=True)
        
        # Use a realistic browser user-agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://stocktwits.com/',
            'Origin': 'https://stocktwits.com'
        }
        
    def fetch_symbol_messages(self, symbol, limit=30, save=True):
        """
        Fetch messages for a specific symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., "AAPL", "MSFT")
        limit : int
            Number of messages to fetch (max 30)
        save : bool
            Whether to save data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing messages or None if API request fails
        """
        endpoint = f"{self.base_url}/streams/symbol/{symbol}.json"
        params = {
            "limit": min(limit, 30),  # StockTwits API has a maximum of 30
            "filter": "all"
        }
        
        try:
            # Add a small random delay to appear more like a human user
            time.sleep(random.uniform(0.5, 2))
            
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get("response", {}).get("status") == 200:
                messages = data.get("messages", [])
                
                if not messages:
                    logger.info(f"No messages found for symbol {symbol}")
                    return None
                
                records = []
                for message in messages:
                    # Extract sentiment if available
                    sentiment = "None"
                    if "entities" in message and "sentiment" in message["entities"]:
                        sentiment = message["entities"]["sentiment"].get("basic", "None")
                    
                    record = {
                        "id": message.get("id"),
                        "text": message.get("body"),
                        "created_at": message.get("created_at"),
                        "user": message.get("user", {}).get("username"),
                        "followers_count": message.get("user", {}).get("followers"),
                        "likes_count": message.get("likes", {}).get("total", 0),
                        "sentiment": sentiment,
                        "symbol": symbol
                    }
                    records.append(record)
                
                df = pd.DataFrame(records)
                if not df.empty:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                
                logger.info(f"Successfully fetched {len(df)} messages for symbol {symbol}")
                
                if save and not df.empty:
                    # Save to CSV
                    filename = f"data/stocktwits/{symbol}_stocktwits_{datetime.now().strftime('%Y%m%d')}.csv"
                    df.to_csv(filename, index=False)
                    logger.info(f"Saved StockTwits data to {filename}")
                
                return df
            else:
                logger.error(f"Failed to fetch StockTwits data. Status: {data.get('response', {}).get('status')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching StockTwits data: {e}")
            
            # If the error is 403 Forbidden, try alternative access method
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 403:
                logger.info("Trying alternative access method via web scraping...")
                return self._fetch_via_scraping(symbol, limit, save)
                
            return None
    
    def _fetch_via_scraping(self, symbol, limit=30, save=True):
        """
        Alternative method to fetch StockTwits data via web scraping
        when the API access is blocked
        """
        import requests
        from bs4 import BeautifulSoup
        
        url = f"https://stocktwits.com/symbol/{symbol}"
        
        try:
            # Add a small random delay to appear more like a human user
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find message containers
            messages = []
            message_containers = soup.find_all('div', class_='st_3LNEQfSg st_1PYr6ciG')
            
            for container in message_containers[:limit]:
                try:
                    # Extract message text
                    message_text_div = container.find('div', class_='st_28bQfzV5')
                    text = message_text_div.get_text(strip=True) if message_text_div else ""
                    
                    # Extract username 
                    username_div = container.find('span', class_='st_1q3ozG2m')
                    username = username_div.get_text(strip=True) if username_div else ""
                    
                    # Try to extract sentiment (this is a bit tricky with scraping)
                    sentiment = "None"
                    sentiment_spans = container.find_all('span', class_='st_3zPu7zM')
                    for span in sentiment_spans:
                        if "Bullish" in span.get_text():
                            sentiment = "Bullish"
                            break
                        elif "Bearish" in span.get_text():
                            sentiment = "Bearish"
                            break
                    
                    # Extract time (will be relative like "2h ago")
                    time_div = container.find('div', class_='st_MVef4nj')
                    time_text = time_div.get_text(strip=True) if time_div else ""
                    
                    # Get current time as a timestamp
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    messages.append({
                        "id": hash(text + username + timestamp),  # Create a unique ID
                        "text": text,
                        "created_at": timestamp,  # Use current time
                        "time_text": time_text,
                        "user": username,
                        "sentiment": sentiment,
                        "symbol": symbol
                    })
                    
                except Exception as e:
                    logger.error(f"Error parsing a message: {e}")
                    continue
            
            if not messages:
                logger.warning(f"No messages found for {symbol} via web scraping")
                return None
                
            df = pd.DataFrame(messages)
            
            if save and not df.empty:
                filename = f"data/stocktwits/{symbol}_stocktwits_scraped_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved scraped StockTwits data to {filename}")
                
            logger.info(f"Successfully scraped {len(df)} messages for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping StockTwits website: {e}")
            return None
    
    def fetch_multiple_symbols(self, symbol_list, limit_per_symbol=30, delay=2):
        """Fetch messages for multiple symbols"""
        result = {}
        for symbol in symbol_list:
            df = self.fetch_symbol_messages(symbol, limit_per_symbol)
            if df is not None and not df.empty:
                result[symbol] = df
                
            # Add delay between requests to avoid rate limiting
            if symbol != symbol_list[-1]:  
                time.sleep(delay + random.uniform(0.5, 2))
                
        return result
    
    def get_sentiment_distribution(self, symbol):
        """Get sentiment distribution for a symbol"""
        df = self.fetch_symbol_messages(symbol, save=False)
        if df is not None and not df.empty:
            sentiment_counts = df['sentiment'].value_counts().to_dict()
            return sentiment_counts
        return {}