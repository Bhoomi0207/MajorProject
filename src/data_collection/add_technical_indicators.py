import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def add_technical_indicators(data, save=True):
    """
    Add technical indicators to price data
    """
    if not isinstance(data, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame")
        return None
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Make sure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns. DataFrame must contain: {required_cols}")
        return None
    
    # Get ticker symbol if available
    ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else 'UNKNOWN'
    logger.info(f"Adding technical indicators for {ticker}")
    
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_StdDev']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_StdDev']
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Average True Range (ATR)
    tr1 = abs(df['High'] - df['Low'])
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    df['TR'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Percentage change
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Daily_Return_Smoothed'] = df['Daily_Return'].rolling(window=5).mean()
    
    # Calculate volatility (standard deviation of returns)
    df['Volatility_5d'] = df['Daily_Return'].rolling(window=5).std()
    df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
    
    # Market momentum indicators
    df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
    
    logger.info(f"Added {len(df.columns) - len(data.columns)} technical indicators")
    
    if save and 'ticker' in df.columns:
        # Save the enhanced data
        ticker = df['ticker'].iloc[0]
        output_file = f"data/historical/{ticker}_enhanced.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced data to {output_file}")
    
    return df

def process_all_tickers(data_dir="data/historical"):
    """
    Process all ticker files in the directory
    """
    # Find all merged historical files
    files = [f for f in os.listdir(data_dir) if f.endswith("_merged_historical.csv")]
    
    if not files:
        logger.warning(f"No merged historical files found in {data_dir}")
        return
    
    logger.info(f"Found {len(files)} files to process")
    
    results = {}
    for file in files:
        try:
            ticker = file.split("_")[0]
            logger.info(f"Processing {ticker}...")
            
            # Load the data
            df = pd.read_csv(os.path.join(data_dir, file))
            
            # Add technical indicators
            enhanced_df = add_technical_indicators(df)
            
            if enhanced_df is not None:
                results[ticker] = enhanced_df
                
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    
    logger.info(f"Successfully processed {len(results)}/{len(files)} files")
    return results

def generate_visualization(ticker, data_dir="data/historical"):
    """
    Generate a visualization of key indicators for a ticker
    """
    # Load the enhanced data
    file_path = os.path.join(data_dir, f"{ticker}_enhanced.csv")
    if not os.path.exists(file_path):
        logger.error(f"Enhanced data file not found: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create a visualization of price, RSI, and MACD
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price chart with Bollinger Bands
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    ax1.plot(df['Date'], df['BB_Upper'], '--', label='Upper BB', color='gray', alpha=0.7)
    ax1.plot(df['Date'], df['BB_Middle'], '--', label='Middle BB', color='gray', alpha=0.7)
    ax1.plot(df['Date'], df['BB_Lower'], '--', label='Lower BB', color='gray', alpha=0.7)
    ax1.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{ticker} Price and Technical Indicators')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2.plot(df['Date'], df['RSI'], color='purple')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(df['Date'], df['RSI'], 70, where=(df['RSI'] >= 70), color='red', alpha=0.3)
    ax2.fill_between(df['Date'], df['RSI'], 30, where=(df['RSI'] <= 30), color='green', alpha=0.3)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # MACD
    ax3.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    ax3.plot(df['Date'], df['MACD_Signal'], label='Signal', color='red')
    ax3.bar(df['Date'], df['MACD_Hist'], label='Histogram', color='gray', alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax3.set_ylabel('MACD')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"data/historical/visualizations/{ticker}_technical.png")
    logger.info(f"Saved visualization to data/historical/visualizations/{ticker}_technical.png")
    plt.close()

if __name__ == "__main__":
    # Create visualization directory if it doesn't exist
    os.makedirs("data/historical/visualizations", exist_ok=True)
    
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process all tickers
    results = process_all_tickers()
    
    # Generate visualizations for a few key tickers
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
        generate_visualization(ticker)
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")