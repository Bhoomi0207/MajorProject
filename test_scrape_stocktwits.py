import sys
import os
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def scrape_stocktwits(symbol):
    """
    Scrape StockTwits messages for a symbol directly from the website
    """
    # Create directory for saving data
    os.makedirs("data/stocktwits", exist_ok=True)
    
    url = f"https://stocktwits.com/symbol/{symbol}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    try:
        logger.info(f"Scraping StockTwits for {symbol}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the message stream container
        stream_container = soup.find('div', {'id': 'messages-container'})
        
        if not stream_container:
            logger.warning(f"Could not find message container for {symbol}")
            return None
        
        # Find all message items
        messages = []
        message_items = stream_container.find_all('div', class_='st_3LNEQfSg')
        
        for item in message_items:
            try:
                # Message text
                message_div = item.find('div', class_='st_28bQfzV5')
                message_text = message_div.get_text() if message_div else "N/A"
                
                # User name
                user_div = item.find('span', class_='st_1q3ozG2m')
                username = user_div.get_text() if user_div else "N/A"
                
                # Time posted
                time_div = item.find('span', class_='st_3QTv-Ni')
                time_text = time_div.get_text() if time_div else "N/A"
                
                # Try to extract sentiment
                sentiment = "None"
                if "Bullish" in str(item):
                    sentiment = "Bullish"
                elif "Bearish" in str(item):
                    sentiment = "Bearish"
                
                messages.append({
                    "symbol": symbol,
                    "text": message_text.strip(),
                    "username": username.strip(),
                    "time": time_text.strip(),
                    "sentiment": sentiment,
                    "scraped_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
            except Exception as e:
                logger.error(f"Error parsing a message: {e}")
                continue
        
        if not messages:
            logger.warning(f"No messages found for {symbol}")
            return None
        
        # Create DataFrame and save
        df = pd.DataFrame(messages)
        filename = f"data/stocktwits/{symbol}_scraped_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        
        logger.info(f"Successfully scraped {len(df)} messages for {symbol}")
        logger.info(f"Data saved to {filename}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error scraping StockTwits for {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Test with popular stocks
    symbols = ["AAPL", "TSLA", "MSFT", "AMZN", "META"]
    
    for symbol in symbols:
        df = scrape_stocktwits(symbol)
        if df is not None:
            print(f"\n=== Sample data for {symbol} ===")
            print(df[["text", "username", "sentiment"]].head(2))
        
        # Add delay between requests
        if symbol != symbols[-1]:
            delay = random.uniform(3, 6)
            logger.info(f"Waiting {delay:.1f} seconds before next request...")
            time.sleep(delay)