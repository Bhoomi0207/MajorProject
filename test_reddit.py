import os
import praw
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

# Get credentials from environment variables
client_id = os.environ.get("REDDIT_CLIENT_ID")
client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
user_agent = os.environ.get("REDDIT_USER_AGENT")

print(f"Client ID: {client_id}")
print(f"Client Secret: {'*' * len(client_secret) if client_secret else None}")
print(f"User Agent: {user_agent}")

try:
    # Initialize the Reddit API client
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    # Test a simple API call
    subreddit = reddit.subreddit("wallstreetbets")
    print(f"Subreddit name: {subreddit.display_name}")
    print(f"Subreddit title: {subreddit.title}")
    print(f"Subreddit subscribers: {subreddit.subscribers}")
    print("API connection successful!")
    
except Exception as e:
    print(f"Error: {e}")