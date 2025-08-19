import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging
import time
import os
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsScraper:
    """
    Class to scrape financial news from various sources
    """
    def __init__(self):
        # Set user agent to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Create data directory if it doesn't exist
        os.makedirs("data/news", exist_ok=True)
        
    def _make_request(self, url):
        """
        Make HTTP request with error handling and random delay
        """
        try:
            # Random delay between requests (0.5-2 seconds)
            time.sleep(random.uniform(0.5, 2))
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            return None
    
    def scrape_yahoo_finance_news(self, ticker, save=True):
        """
        Scrape news from Yahoo Finance for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing news articles
        """
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        response = self._make_request(url)
        
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        try:
            # Find all news articles in the page
            articles = soup.find_all('div', {'class': 'Ov(h)'})
            
            for article in articles:
                title_element = article.find('h3')
                if not title_element:
                    continue
                    
                title = title_element.text.strip()
                
                # Get link
                link_element = title_element.find('a')
                link = ''
                if link_element and 'href' in link_element.attrs:
                    link = "https://finance.yahoo.com" + link_element['href'] if link_element['href'].startswith('/') else link_element['href']
                
                # Get source and time
                source_time = article.find('div', {'class': 'C(#959595)'})
                source = ""
                published_time = ""
                
                if source_time:
                    source_time_text = source_time.text.strip()
                    if '·' in source_time_text:
                        source, published_time = source_time_text.split('·', 1)
                        source = source.strip()
                        published_time = published_time.strip()
                    else:
                        source = source_time_text
                
                # Get summary if available
                summary = ""
                summary_element = article.find('p')
                if summary_element:
                    summary = summary_element.text.strip()
                
                news_items.append({
                    'ticker': ticker,
                    'title': title,
                    'source': source,
                    'published_time': published_time,
                    'summary': summary,
                    'link': link,
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            logger.info(f"Successfully scraped {len(news_items)} news articles for {ticker}")
            
            if not news_items:
                return None
                
            df = pd.DataFrame(news_items)
            
            if save and not df.empty:
                # Save to CSV
                filename = f"data/news/{ticker}_yahoo_news_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved news data to {filename}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance news for {ticker}: {e}")
            return None
    
    def scrape_seeking_alpha(self, ticker, save=True):
        """
        Scrape news from Seeking Alpha for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing news articles
        """
        url = f"https://seekingalpha.com/symbol/{ticker}/news"
        response = self._make_request(url)
        
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        try:
            # Find all news articles
            article_containers = soup.find_all('div', {'data-test-id': 'post-list-item'})
            
            for container in article_containers:
                # Get title
                title_element = container.find('a', {'data-test-id': 'post-list-item-title'})
                if not title_element:
                    continue
                
                title = title_element.text.strip()
                link = "https://seekingalpha.com" + title_element['href'] if title_element.has_attr('href') else ""
                
                # Get author
                author = ""
                author_element = container.find('a', {'data-test-id': 'post-list-item-author'})
                if author_element:
                    author = author_element.text.strip()
                
                # Get date
                date = ""
                date_element = container.find('span', {'data-test-id': 'post-list-item-date'})
                if date_element:
                    date = date_element.text.strip()
                
                # Get summary
                summary = ""
                summary_element = container.find('p')
                if summary_element:
                    summary = summary_element.text.strip()
                
                news_items.append({
                    'ticker': ticker,
                    'title': title,
                    'author': author,
                    'date': date,
                    'summary': summary,
                    'link': link,
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            logger.info(f"Successfully scraped {len(news_items)} Seeking Alpha articles for {ticker}")
            
            if not news_items:
                return None
                
            df = pd.DataFrame(news_items)
            
            if save and not df.empty:
                # Save to CSV
                filename = f"data/news/{ticker}_seeking_alpha_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved Seeking Alpha data to {filename}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Seeking Alpha for {ticker}: {e}")
            return None

    def scrape_marketwatch(self, ticker, save=True):
        """
        Scrape news from MarketWatch for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing news articles
        """
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
        response = self._make_request(url)
        
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        try:
            # Find the news section
            news_section = soup.find('div', {'class': 'tab__content'})
            if not news_section:
                logger.warning(f"News section not found for {ticker} on MarketWatch")
                return None
                
            # Find all news articles
            articles = news_section.find_all('div', {'class': 'article__content'})
            
            for article in articles:
                # Get title
                title_element = article.find('h3', {'class': 'article__headline'})
                if not title_element:
                    continue
                    
                title_link = title_element.find('a')
                if not title_link:
                    continue
                    
                title = title_link.text.strip()
                link = title_link['href'] if 'href' in title_link.attrs else ""
                
                # Get date/time
                time_element = article.find('span', {'class': 'article__timestamp'})
                published_time = time_element.text.strip() if time_element else ""
                
                # Get summary
                summary = ""
                summary_element = article.find('p', {'class': 'article__summary'})
                if summary_element:
                    summary = summary_element.text.strip()
                
                news_items.append({
                    'ticker': ticker,
                    'title': title,
                    'published_time': published_time,
                    'summary': summary,
                    'link': link,
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            logger.info(f"Successfully scraped {len(news_items)} MarketWatch articles for {ticker}")
            
            if not news_items:
                return None
                
            df = pd.DataFrame(news_items)
            
            if save and not df.empty:
                # Save to CSV
                filename = f"data/news/{ticker}_marketwatch_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved MarketWatch data to {filename}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error scraping MarketWatch for {ticker}: {e}")
            return None
    
    def scrape_reddit_wallstreetbets(self, query, limit=25, save=True):
        """
        Scrape posts from Reddit's r/wallstreetbets subreddit
        Note: This uses old.reddit.com which is more scraper-friendly
        
        Parameters:
        -----------
        query : str
            Search query (typically a ticker symbol)
        limit : int
            Maximum number of results to fetch
        save : bool
            Whether to save the data to CSV
            
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing Reddit posts
        """
        url = f"https://old.reddit.com/r/wallstreetbets/search?q={query}&restrict_sr=on"
        response = self._make_request(url)
        
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        posts = []
        
        try:
            # Find all search results
            results = soup.find_all('div', {'class': 'search-result-link'})
            count = 0
            
            for result in results:
                if count >= limit:
                    break
                    
                # Get post title and link
                title_element = result.find('a', {'class': 'search-title'})
                if not title_element:
                    continue
                
                title = title_element.text.strip()
                link = "https://www.reddit.com" + title_element['href'] if title_element.has_attr('href') else ""
                
                # Get post details
                details = result.find('span', {'class': 'search-result-meta'})
                posted_by = ""
                subreddit = ""
                date = ""
                
                if details:
                    details_text = details.text.strip()
                    parts = details_text.split('in')
                    
                    if len(parts) > 1:
                        posted_info = parts[0].strip()
                        subreddit_info = parts[1].strip()
                        
                        # Extract username
                        if 'by' in posted_info:
                            posted_by = posted_info.split('by')[1].strip()
                        
                        # Extract date - typically in format "X days/months/years ago"
                        date_parts = posted_info.split()
                        if len(date_parts) >= 3 and 'ago' in date_parts:
                            date = ' '.join(date_parts[-3:])
                            
                        # Extract subreddit
                        subreddit = subreddit_info
                
                posts.append({
                    'query': query,
                    'title': title,
                    'posted_by': posted_by,
                    'date': date,
                    'subreddit': subreddit,
                    'link': link,
                    'scraped_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                count += 1
            
            logger.info(f"Successfully scraped {len(posts)} Reddit posts for {query}")
            
            if not posts:
                return None
                
            df = pd.DataFrame(posts)
            
            if save and not df.empty:
                # Save to CSV
                filename = f"data/news/{query}_reddit_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved Reddit data to {filename}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error scraping Reddit for {query}: {e}")
            return None
    
    def scrape_all_sources(self, ticker):
        """
        Scrape all news sources for a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        dict
            Dictionary with source names as keys and DataFrames as values
        """
        results = {}
        
        # Yahoo Finance
        yahoo_df = self.scrape_yahoo_finance_news(ticker)
        if yahoo_df is not None and not yahoo_df.empty:
            results['yahoo'] = yahoo_df
        
        # Seeking Alpha
        seeking_alpha_df = self.scrape_seeking_alpha(ticker)
        if seeking_alpha_df is not None and not seeking_alpha_df.empty:
            results['seeking_alpha'] = seeking_alpha_df
        
        # MarketWatch
        marketwatch_df = self.scrape_marketwatch(ticker)
        if marketwatch_df is not None and not marketwatch_df.empty:
            results['marketwatch'] = marketwatch_df
        
        # Reddit
        reddit_df = self.scrape_reddit_wallstreetbets(ticker)
        if reddit_df is not None and not reddit_df.empty:
            results['reddit'] = reddit_df
        
        return results