import os
import pandas as pd
import yfinance as yf

# Folder containing your historical CSVs
DATA_DIR = "data/historical"

# Function to fetch stock data for a given ticker
def get_stock_data(symbol, start="2025-03-01", end="2025-07-31"):
    """
    Fetch stock price data from Yahoo Finance.
    :param symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
    :param start: Start date (YYYY-MM-DD)
    :param end: End date (YYYY-MM-DD)
    :return: Pandas DataFrame with stock data
    """
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Find all tickers from files in data/historical
    tickers = []
    for file in os.listdir(DATA_DIR):
        if file.endswith("_merged_historical.csv"):
            tickers.append(file.split("_")[0])

    print(f"Found {len(tickers)} tickers: {', '.join(tickers)}")

    # Fetch & save updated data for each ticker
    for ticker in tickers:
        print(f"⏳ Fetching {ticker}...")
        df = get_stock_data(ticker, start="2025-03-16", end="2025-07-31")
        if df is not None and not df.empty:
            out_path = os.path.join(DATA_DIR, f"{ticker}_latest.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved {ticker} → {out_path}")
        else:
            print(f"⚠️ No data retrieved for {ticker}")
