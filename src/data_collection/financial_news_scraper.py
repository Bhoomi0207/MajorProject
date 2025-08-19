import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import logging
from datetime import datetime
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialNewsScraper:
    """
    Class to scrape financial news from various sources
    """
    def __init__(self):
        # Set up headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept