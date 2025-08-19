import os
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get API key
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Test
print("API Key Loaded:", api_key)


# Load your API key (replace with your real one if not using env)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "your_api_key_here")

# Initialize Alpha Vantage TimeSeries
ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format='pandas')

def get_stock_data(symbol, interval='1min'):
    """
    Fetch stock price data from Alpha Vantage
    :param symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
    :param interval: Time interval ('1min', '5min', '15min', '30min', '60min')
    :return: Pandas DataFrame with stock data
    """
    try:
        data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize='compact')
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

if __name__ == "__main__":
    # Example: Fetch Apple stock data
    df = get_stock_data("AAPL", interval="5min")
    if df is not None:
        print(df.head())
        df.to_csv("data/historical/AAPL_latest.csv", index=False)
        print("Data saved to data/historical/AAPL_latest.csv")



