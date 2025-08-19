"""
Terminal-based model accuracy evaluation script.
This script evaluates the accuracy of prediction models using backtesting
without having to run the full dashboard.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.predictive_models.price_predictor import StockPricePredictor
from src.predictive_models.lstm_model import LSTMStockPredictor
from src.predictive_models.ml_models import MLStockPredictor
from dashboard.data_loader import StockDataLoader  # Fixed import - using the correct class name

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate model accuracy from terminal')
    
    parser.add_argument('--ticker', type=str, default='AAPL', 
                        help='Stock ticker symbol (default: AAPL)')
    
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'random_forest', 'xgboost'],
                        help='Model type to evaluate (default: lstm)')
    
    parser.add_argument('--days', type=int, default=5,
                        help='Prediction horizon in days (default: 5)')
    
    parser.add_argument('--periods', type=int, default=3,
                        help='Number of backtest periods (default: 3)')
    
    parser.add_argument('--train-days', type=int, default=60,
                        help='Number of days to use for training (default: 60)')
    
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to files')
    
    return parser.parse_args()

def evaluate_model_accuracy(
    ticker="AAPL", 
    model="lstm", 
    days=5, 
    periods=3, 
    train_days=60, 
    save_results=False
):
    """Reusable function to evaluate model accuracy (used in Streamlit or CLI)."""
    data_loader = StockDataLoader()
    predictor = StockPricePredictor(
        data_dir='data',
        models_dir='models',
        results_dir='results'
    )

    # Load stock data
    df = data_loader.load_stock_data(ticker)
    if df is None or df.empty:
        return {"error": f"Could not load data for {ticker}"}

    processed_df = predictor.preprocess_data(df)
    if processed_df is None or processed_df.empty:
        return {"error": "Failed to preprocess data"}

    # Model mapping
    model_map = {
        "lstm": "lstm_Close",
        "random_forest": "random_forest_regression_Close",
        "xgboost": "xgboost_regression_Close"
    }
    model_name = model_map.get(model)

    # Train if necessary
    if model_name not in predictor.models:
        if model == "lstm":
            predictor.train_lstm_model(
                processed_df, target_column="Close",
                sequence_length=10, epochs=50, batch_size=32
            )
            model_instance = predictor.models.get("lstm_Close")
        else:
            predictor.train_ml_model(
                processed_df, target_column="Close",
                model_type=model, task_type="regression"
            )
            model_instance = predictor.models.get(f"{model}_regression_Close")
    else:
        model_instance = predictor.models.get(model_name)

    if model_instance is None:
        return {"error": f"Model {model} failed to load/train"}

    # Run backtesting
    backtest_results = predictor.evaluator.backtest_predictions(
        processed_df, model_instance,
        prediction_days=days, target_column="Close",
        backtest_periods=periods,
        test_size=train_days / len(processed_df),
        save_results=save_results
    )

    # 🆕 Wrap results nicely
    return {
        "ticker": ticker,
        "model": model,
        "days": days,
        "periods": periods,
        "metrics": backtest_results  # keep original details here
    }

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    print(f"\nEvaluating {args.model.upper()} model accuracy for {args.ticker}...")
    print(f"Prediction horizon: {args.days} days, Backtest periods: {args.periods}\n")
    
    # Initialize components
    data_loader = StockDataLoader()  # Fixed: Using the correct class name
    predictor = StockPricePredictor(
        data_dir='data',
        models_dir='models',
        results_dir='results'
    )
    
    try:
        # Load stock data
        print(f"Loading data for {args.ticker}...")
        df = data_loader.load_stock_data(args.ticker)
        
        if df is None or df.empty:
            print(f"Error: Could not load data for {args.ticker}")
            return
        
        # Preprocess data
        print("Preprocessing data...")
        processed_df = predictor.preprocess_data(df)
        
        if processed_df is None or processed_df.empty:
            print("Error: Failed to preprocess data")
            return
        
        # Map model type to correct model instance
        model_map = {
            "lstm": "lstm_Close",
            "random_forest": "random_forest_regression_Close",
            "xgboost": "xgboost_regression_Close"
        }
        
        model_name = model_map.get(args.model)
        
        # Check if we have the model or need to train it
        if model_name not in predictor.models:
            print(f"Model {args.model} not found. Training a new model...")
            
            if args.model == "lstm":
                predictor.train_lstm_model(
                    processed_df,
                    target_column='Close',
                    sequence_length=10,
                    epochs=50,
                    batch_size=32
                )
                model = predictor.models.get("lstm_Close")
            else:
                # For ML models (Random Forest, XGBoost)
                predictor.train_ml_model(
                    processed_df,
                    target_column='Close',
                    model_type=args.model,
                    task_type='regression'
                )
                model = predictor.models.get(f"{args.model}_regression_Close")
        else:
            model = predictor.models.get(model_name)
        
        if model is None:
            print(f"Error: Failed to load or train model")
            return
            
        # Run backtesting
        print(f"Running backtesting with {args.periods} periods, {args.days} days prediction horizon...")
        
        backtest_results = predictor.evaluator.backtest_predictions(
            processed_df,
            model,
            prediction_days=args.days,
            target_column='Close',
            backtest_periods=args.periods,
            test_size=args.train_days / len(processed_df),
            save_results=args.save_results
        )
        
        if not backtest_results:
            print("Error: Backtesting failed to produce results")
            return
            
        # Display summary
        print("\nEvaluation completed successfully!")
        if args.save_results:
            print("Results saved to the 'results' directory")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
    
if __name__ == "__main__":
    main()