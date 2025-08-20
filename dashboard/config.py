"""
Configuration settings for the dashboard application
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data directories
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, "historical")
NEWS_DATA_DIR = os.path.join(DATA_DIR, "news")

# API Keys
NEWS_API_KEY = "a262dd9b69d047e4a8073653d022a3a0"  # Consider moving to environment variables

# App settings
HOST = "localhost"
PORT = 8050
DEBUG = True

# Visualization settings
DEFAULT_PLOT_THEME = "plotly_white"
COLOR_SCHEME = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffbb33",
    "info": "#17becf"
}

# Time periods
DEFAULT_TIMEFRAME = 365  # days
MOVING_AVERAGES = [20, 50, 200]  # days
VOLATILITY_WINDOW = 20  # days

# Dashboard settings
MAX_NEWS_ARTICLES = 20
DEFAULT_TICKER = "AAPL"  # Default stock to show