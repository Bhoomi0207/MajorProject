import praw
import pandas as pd
import datetime
import time
from textblob import TextBlob
import configparser
import os
import argparse

from dotenv import load_dotenv
load_dotenv()


class RedditScraper:
    def __init__(self, client_id=None, client_secret=None, user_agent=None, username='', password=''):
        """Initialize the Reddit scraper with your API credentials."""

        self.reddit = praw.Reddit(
            client_id=client_id or os.getenv("REDDIT_CLIENT_ID"),
            client_secret=client_secret or os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=user_agent or os.getenv("REDDIT_USER_AGENT"),
            username=username,
            password=password
        )

        
    def get_subreddit_comments(self, subreddit_name, limit=1000, time_filter='week'):
        """
        Scrape comments from a specified subreddit.
        
        Parameters:
        - subreddit_name: Name of the subreddit (without r/)
        - limit: Maximum number of comments to retrieve
        - time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
        
        Returns:
        - DataFrame with comments and metadata
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        
        comments_data = []
        
        # Get top posts from the subreddit
        for submission in subreddit.top(time_filter=time_filter, limit=100):
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            # Process each comment
            for comment in submission.comments.list():
                if len(comments_data) >= limit:
                    break
                    
                # Analyze sentiment
                sentiment = self.analyze_sentiment(comment.body)
                
                comments_data.append({
                    'comment_id': comment.id,
                    'post_id': submission.id,
                    'post_title': submission.title,
                    'subreddit': subreddit_name,
                    'author': str(comment.author),
                    'comment_text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.datetime.fromtimestamp(comment.created_utc),
                    'sentiment_polarity': sentiment.polarity,
                    'sentiment_subjectivity': sentiment.subjectivity
                })
            
            if len(comments_data) >= limit:
                break
                
        return pd.DataFrame(comments_data)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob.
        
        Returns:
        - sentiment object with polarity (-1 to 1) and subjectivity (0 to 1)
        """
        return TextBlob(text).sentiment
    
    def scrape_multiple_subreddits(self, subreddit_list, comments_per_sub=500, time_filter='week'):
        """
        Scrape comments from multiple subreddits.
        
        Parameters:
        - subreddit_list: List of subreddit names
        - comments_per_sub: Number of comments to retrieve per subreddit
        - time_filter: Time filter for posts
        
        Returns:
        - Combined DataFrame with all comments
        """
        all_comments = pd.DataFrame()
        
        for subreddit in subreddit_list:
            print(f"Scraping r/{subreddit}...")
            try:
                df = self.get_subreddit_comments(subreddit, limit=comments_per_sub, time_filter=time_filter)
                all_comments = pd.concat([all_comments, df])
                print(f"  - Retrieved {len(df)} comments")
                # Sleep to respect rate limits
                time.sleep(2)
            except Exception as e:
                print(f"Error scraping r/{subreddit}: {e}")
        
        return all_comments
    
    def save_data(self, dataframe, output_file='data/historical/reddit_financial_comments.csv'):
        """Save scraped data to a CSV file."""
        dataframe.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Scrape Reddit comments from financial subreddits')
    parser.add_argument('--config', type=str, default='config.ini', help='Path to config file')
    parser.add_argument('--output', type=str, default='data/historical/reddit_financial_comments.csv', help='Output file path')
    parser.add_argument('--limit', type=int, default=500, help='Comments per subreddit')
    parser.add_argument('--timefilter', type=str, default='week', help='Time filter (hour, day, week, month, year, all)')
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Creating sample config file at {args.config}")
        with open(args.config, 'w') as f:
            f.write("""[API_CREDENTIALS]
client_id = YOUR_CLIENT_ID
client_secret = YOUR_CLIENT_SECRET
user_agent = python:financial-sentiment-analyzer:v1.0 (by /u/YOUR_USERNAME)
username = YOUR_REDDIT_USERNAME
password = YOUR_REDDIT_PASSWORD

[SUBREDDITS]
# Comma-separated list of financial subreddits
subreddits = wallstreetbets, investing, stocks, finance, personalfinance, cryptocurrency, CryptoMarkets, Bitcoin
""")
        print(f"Please edit {args.config} with your Reddit API credentials")
        return
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # Initialize scraper
    scraper = RedditScraper(
        client_id=config['API_CREDENTIALS']['client_id'],
        client_secret=config['API_CREDENTIALS']['client_secret'],
        user_agent=config['API_CREDENTIALS']['user_agent'],
        username=config['API_CREDENTIALS'].get('username', ''),
        password=config['API_CREDENTIALS'].get('password', '')
    )
    
    # Get list of subreddits
    subreddits = [s.strip() for s in config['SUBREDDITS']['subreddits'].split(',')]
    
    print(f"Starting to scrape {len(subreddits)} subreddits...")
    data = scraper.scrape_multiple_subreddits(subreddits, args.limit, args.timefilter)
    
    # Save the data
    scraper.save_data(data, args.output)
    
    # Print sentiment summary
    print("\nSentiment Summary:")
    summary = data.groupby('subreddit')[['sentiment_polarity', 'sentiment_subjectivity']].mean()
    print(summary)
    
    # Find most positive and negative comments
    most_positive = data.loc[data['sentiment_polarity'].idxmax()]
    most_negative = data.loc[data['sentiment_polarity'].idxmin()]
    
    print("\nMost positive comment:")
    print(f"Subreddit: r/{most_positive['subreddit']}")
    print(f"Score: {most_positive['score']}")
    print(f"Sentiment: {float(most_positive['sentiment_polarity']):.2f}")
    print(f"Comment: {most_positive['comment_text'][:100]}...")

    print("\nMost negative comment:")
    print(f"Subreddit: r/{most_negative['subreddit']}")
    print(f"Score: {most_negative['score']}")
    print(f"Sentiment: {float(most_negative['sentiment_polarity']):.2f}")
    print(f"Comment: {most_negative['comment_text'][:100]}...")



if __name__ == "__main__":
    main()