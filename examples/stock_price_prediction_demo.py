"""
Example script demonstrating stock price prediction with various models.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local modules
from src.predictive_models.price_predictor import StockPricePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to demonstrate stock price prediction
    """
    # Initialize the stock price predictor
    predictor = StockPricePredictor(
        data_dir='data',
        models_dir='models',
        results_dir='results'
    )
    
    # Define ticker to analyze
    ticker = 'BA'  # Boeing
    
    # Run complete forecasting pipeline
    logger.info(f"Starting forecasting pipeline for {ticker}")
    
    results = predictor.run_forecasting_pipeline(
        ticker=ticker,
        train_test_split=0.8,  # Use 80% of data for training
        prediction_days=10,    # Predict 10 days ahead
        save_models=True,
        save_results=True
    )
    
    if results:
        # Print summary of results
        logger.info(f"Forecasting pipeline completed for {ticker}")
        logger.info(f"Data shape: {results['data_shape']}")
        logger.info(f"Trained models: {results['models']}")
        
        # Display price predictions
        if 'price_predictions' in results and results['price_predictions'] is not None:
            price_predictions = results['price_predictions']
            logger.info(f"Price predictions for next {len(price_predictions)} days:")
            logger.info(price_predictions[['Close']])
            
            # Plot price predictions
            plt.figure(figsize=(12, 6))
            plt.plot(price_predictions['Close'], marker='o', linestyle='-', linewidth=2)
            plt.title(f"{ticker} Price Predictions")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    else:
        logger.error(f"Failed to complete forecasting pipeline for {ticker}")

def compare_multiple_stocks():
    """
    Compare predictions for multiple stocks
    """
    # Initialize the stock price predictor
    predictor = StockPricePredictor(
        data_dir='data',
        models_dir='models',
        results_dir='results'
    )
    
    # Define tickers to analyze
    tickers = ['BA', 'MSFT', 'AAPL']  # Boeing, Microsoft, Apple
    
    predictions = {}
    directions = {}
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}")
        
        # Load and preprocess data
        df = predictor.load_stock_data(ticker)
        if df is None:
            logger.error(f"Failed to load data for {ticker}")
            continue
            
        processed_df = predictor.preprocess_data(df)
        
        # Split data
        train_size = int(len(processed_df) * 0.8)
        train_df = processed_df.iloc[:train_size]
        test_df = processed_df.iloc[train_size:]
        
        # Train LSTM model for price prediction
        predictor.train_lstm_model(
            train_df,
            target_column='Close',
            sequence_length=10,
            epochs=50,  # Reduced for demonstration
            batch_size=32
        )
        
        # Train random forest for direction prediction
        predictor.train_ml_model(
            train_df,
            target_column='Target_Direction_1d',
            model_type='random_forest',
            task_type='classification'
        )
        
        # Make predictions
        latest_data = processed_df.iloc[-100:]  # Use last 100 data points
        
        price_pred = predictor.predict_price(
            latest_data,
            steps_ahead=5  # 5-day forecast
        )
        
        dir_pred = predictor.predict_direction(latest_data)
        
        # Store predictions
        if price_pred is not None:
            predictions[ticker] = price_pred['Close']
        
        if dir_pred is not None:
            # Get the latest direction prediction
            directions[ticker] = 'Up' if dir_pred[-1] == 1 else 'Down'
    
    # Plot price predictions for all stocks
    if predictions:
        plt.figure(figsize=(14, 7))
        
        for ticker, pred in predictions.items():
            plt.plot(pred, marker='o', linestyle='-', linewidth=2, label=f"{ticker}")
        
        plt.title("Stock Price Predictions Comparison")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print direction predictions
        logger.info("Price direction predictions:")
        for ticker, direction in directions.items():
            logger.info(f"{ticker}: {direction}")

if __name__ == "__main__":
    # Run single stock example
    main()
    
    # Uncomment to run multiple stock comparison
    # compare_multiple_stocks()