import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Secrets
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Yahoo Finance Settings
STOCKS_TO_TRACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "BAC", "GS"
]

# Sector mappings for tracked companies
SECTOR_MAPPING = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "TSLA": "Automotive",
    "NVDA": "Technology",
    "JPM": "Financial Services",
    "BAC": "Financial Services",
    "GS": "Financial Services",
}

# Path for saving data
DATA_PATH = os.path.join("data")