"""
Script to run data preprocessing for stock sentiment analysis
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.data_preprocessor import StockDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess data for stock sentiment analysis")
    parser.add_argument("--tickers", type=str, 
                      default="AAPL,MSFT,GOOGL,AMZN,META",
                      help="Comma-separated list of stock tickers to process")
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Base directory for data files")
    parser.add_argument("--prepare-ml", action="store_true",
                      help="Prepare machine learning datasets")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Parse ticker list
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    logger.info(f"Preparing to preprocess data for {len(tickers)} tickers: {tickers}")
    
    # Initialize the preprocessor
    # Initialize the preprocessor
    preprocessor = StockDataPreprocessor(data_dir=args.data_dir)
    
    # Process each ticker
    results = preprocessor.process_multiple_tickers(tickers)
    
    # Print summary
    successful = sum(1 for data in results.values() if data is not None)
    logger.info("\n===== PREPROCESSING SUMMARY =====")
    logger.info(f"Processed {len(tickers)} tickers")
    logger.info(f"Successfully preprocessed: {successful}/{len(tickers)}")
    
    # Display details for each ticker
    for ticker, data in results.items():
        if data is not None:
            sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
            logger.info(f"  {ticker}: {len(data)} rows, {len(data.columns)} columns, {len(sentiment_cols)} sentiment features")
        else:
            logger.info(f"  {ticker}: Failed to preprocess")
    
    # Prepare ML datasets if requested
    if args.prepare_ml:
        logger.info("\nPreparing machine learning datasets...")
        
        ml_results = {}
        for ticker in tickers:
            if results.get(ticker) is not None:
                try:
                    logger.info(f"Preparing ML dataset for {ticker}...")
                    ml_data = preprocessor.prepare_ml_dataset(ticker)
                    
                    if ml_data is not None:
                        ml_results[ticker] = {
                            'X_train_shape': ml_data['X_train'].shape,
                            'X_test_shape': ml_data['X_test'].shape,
                            'features': len(ml_data['features']),
                            'targets': ml_data['targets']
                        }
                        
                        logger.info(f"  {ticker}: Created ML dataset with {ml_data['X_train'].shape[0]} training samples, " +
                                   f"{ml_data['X_test'].shape[0]} test samples, and {len(ml_data['features'])} features")
                    else:
                        logger.warning(f"  {ticker}: Failed to create ML dataset")
                
                except Exception as e:
                    logger.error(f"Error preparing ML dataset for {ticker}: {e}")
    
    logger.info("==============================")
    

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")