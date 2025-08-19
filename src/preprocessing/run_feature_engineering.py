"""
Script to run feature engineering for stock sentiment analysis
"""
import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
import joblib

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.preprocessing.feature_engineering import StockFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run feature engineering for stock sentiment analysis")
    parser.add_argument("--tickers", type=str, 
                      default="AAPL,MSFT,GOOGL,AMZN,META",
                      help="Comma-separated list of stock tickers to process")
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Base directory for data files")
    parser.add_argument("--prepare-ml", action="store_true",
                      help="Prepare machine learning datasets")
    parser.add_argument("--target", type=str, default="Target_Direction_1d",
                      help="Target variable to predict")
    parser.add_argument("--test-size", type=float, default=0.2,
                      help="Proportion of data to use for testing")
    parser.add_argument("--save-model", action="store_true",
                      help="Save scaler and feature list")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Parse ticker list
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    logger.info(f"Preparing to process {len(tickers)} tickers: {tickers}")
    
    # Setup directories
    processed_dir = os.path.join(args.data_dir, "processed")
    features_dir = os.path.join(args.data_dir, "features")
    ml_dir = os.path.join(args.data_dir, "ml_datasets")
    
    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(ml_dir, exist_ok=True)
    
    # Initialize the feature engineer
    feature_engineer = StockFeatureEngineer()
    
    # Process each ticker
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}...")
            
            # Load preprocessed data
            preprocessed_file = os.path.join(processed_dir, f"{ticker}_preprocessed.csv")
            
            if not os.path.exists(preprocessed_file):
                logger.warning(f"Preprocessed file not found for {ticker}, skipping")
                continue
            
            # Load the data
            df = pd.read_csv(preprocessed_file)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Run feature engineering
            featured_df = feature_engineer.run_complete_feature_engineering(df)
            
            # Save featured data
            featured_file = os.path.join(features_dir, f"{ticker}_featured.csv")
            featured_df.to_csv(featured_file, index=False)
            logger.info(f"Saved featured data to {featured_file}")
            
            # Prepare ML dataset if requested
            if args.prepare_ml:
                ml_data = feature_engineer.prepare_ml_dataset(
                    featured_df, 
                    test_size=args.test_size,
                    target_column=args.target
                )
                
                if ml_data:
                    # Save ML data
                    ml_file = os.path.join(ml_dir, f"{ticker}_ml_data.joblib")
                    joblib.dump(ml_data, ml_file)
                    logger.info(f"Saved ML dataset to {ml_file}")
                    
                    # Save scaler and feature list if requested
                    if args.save_model:
                        from sklearn.preprocessing import StandardScaler
                        
                        # Fit scaler on training data
                        scaler = StandardScaler()
                        scaler.fit(ml_data['X_train'])
                        
                        # Save scaler
                        scaler_file = os.path.join(ml_dir, f"{ticker}_scaler.joblib")
                        joblib.dump(scaler, scaler_file)
                        
                        # Save feature list
                        feature_file = os.path.join(ml_dir, f"{ticker}_features.joblib")
                        joblib.dump(ml_data['feature_names'], feature_file)
                        
                        logger.info(f"Saved scaler and feature list for {ticker}")
                
                # Store results
                results[ticker] = {
                    'features': len(ml_data['feature_names']) if ml_data else 0,
                    'train_samples': len(ml_data['X_train']) if ml_data else 0,
                    'test_samples': len(ml_data['X_test']) if ml_data else 0
                }
            else:
                # Store simple results
                results[ticker] = {
                    'features': len(featured_df.columns) - len(df.columns),
                    'samples': len(featured_df)
                }
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    # Print summary
    logger.info("\n===== FEATURE ENGINEERING SUMMARY =====")
    logger.info(f"Processed {len(tickers)} tickers")
    
    successful = sum(1 for r in results.values() if 'error' not in r)
    logger.info(f"Successfully processed: {successful}/{len(tickers)}")
    
    for ticker, result in results.items():
        if 'error' in result:
            logger.info(f"  {ticker}: Failed - {result['error']}")
        elif args.prepare_ml:
            logger.info(f"  {ticker}: {result['features']} features, {result['train_samples']} train samples, {result['test_samples']} test samples")
        else:
            logger.info(f"  {ticker}: Added {result['features']} features, {result['samples']} samples total")
    
    logger.info("=======================================")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Script completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")