"""
Main interface for stock price prediction.
"""
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle

# Local imports
from src.predictive_models.lstm_model import LSTMStockPredictor
from src.predictive_models.ml_models import MLStockPredictor
from src.predictive_models.model_evaluator import ModelEvaluator
from src.preprocessing.feature_engineering import StockFeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class StockPricePredictor:
    """
    Main class for stock price prediction
    """
    def __init__(self, data_dir='data', models_dir='models', results_dir='results'):
        """
        Initialize stock price predictor
        
        Parameters:
        -----------
        data_dir : str
            Directory for input data
        models_dir : str
            Directory for model storage
        results_dir : str
            Directory for results storage
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        for directory in [data_dir, models_dir, results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize components
        self.feature_engineer = StockFeatureEngineer()
        self.evaluator = ModelEvaluator(output_dir=results_dir)
        
        # Dictionary to store models
        self.models = {}
    
    def load_stock_data(self, ticker, file_path=None):
        """
        Load stock data for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        file_path : str, optional
            Custom file path to load data from
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stock data
        """
        if file_path:
            data_path = file_path
        else:
            # Look for data in default locations
            potential_paths = [
                os.path.join(self.data_dir, 'stocks', f'{ticker}_data.csv'),
                os.path.join(self.data_dir, 'historical', f'{ticker}_merged_historical.csv'),
                os.path.join(self.data_dir, f'{ticker}_data.csv')
            ]
            
            data_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if not data_path:
                logger.error(f"No data file found for ticker {ticker}")
                return None
        
        # Load data
        try:
            df = pd.read_csv(data_path)
            
            # Check if date column exists and convert to datetime
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            logger.info(f"Loaded data for {ticker} with {len(df)} rows from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None
    
    def preprocess_data(self, df, include_sentiment=True):
        """
        Preprocess data for model training
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        include_sentiment : bool
            Whether to include sentiment features
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Make sure we're only working with numeric columns for model training
        # Check if 'symbol' or ticker column exists and remove it
        columns_to_drop = []
        for col in result_df.columns:
            if col.lower() in ['symbol', 'ticker'] or result_df[col].dtype == 'object':
                columns_to_drop.append(col)
        
        if columns_to_drop:
            logger.info(f"Removing non-numeric columns: {columns_to_drop}")
            result_df = result_df.drop(columns=columns_to_drop)
            
        # Run feature engineering
        result_df = self.feature_engineer.add_all_features(
            result_df, 
            include_sentiment=include_sentiment
        )
        
        # Clean up
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.fillna(method='ffill')
        result_df = result_df.fillna(method='bfill')
        
        # If there are still NaN values after filling, replace with 0 instead of dropping
        if result_df.isna().any().any():
            logger.warning(f"Still have NaN values after ffill/bfill, filling with 0")
            result_df = result_df.fillna(0)
        
        # Check if we have data after preprocessing
        if result_df.empty:
            logger.error("Preprocessed DataFrame is empty - returning original with basic cleaning")
            # If result is empty, return at least the original data with basic cleaning
            basic_df = df.copy()
            basic_df = basic_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
            return basic_df
        
        logger.info(f"Preprocessed data with {len(result_df)} rows and {len(result_df.columns)} columns")
        
        return result_df
    
    def train_lstm_model(self, df, target_column='Close', sequence_length=10, epochs=100, 
                         batch_size=32, save_model=True):
        """
        Train LSTM model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with preprocessed data
        target_column : str
            Column to predict
        sequence_length : int
            Number of time steps to look back
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        save_model : bool
            Whether to save the model
            
        Returns:
        --------
        LSTMStockPredictor
            Trained LSTM model
        """
        # Create model directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Create model path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"lstm_model_{timestamp}.h5")
        
        # Initialize LSTM model
        lstm_model = LSTMStockPredictor(
            sequence_length=sequence_length,
            model_path=model_path if save_model else None
        )
        
        # Train model
        history = lstm_model.fit(
            df,
            target_column=target_column,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Store model
        model_name = f"lstm_{target_column}"
        self.models[model_name] = lstm_model
        
        logger.info(f"Trained LSTM model for {target_column} prediction")
        
        return lstm_model
    
    def train_ml_model(self, df, target_column='Target_Direction_1d', model_type='random_forest', 
                       task_type='classification', test_size=0.2, save_model=True):
        """
        Train machine learning model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with preprocessed data
        target_column : str
            Column to predict
        model_type : str
            Type of model to use ('random_forest', 'xgboost', 'svm', 'linear')
        task_type : str
            Type of task ('regression' or 'classification')
        test_size : float
            Proportion of data to use for testing
        save_model : bool
            Whether to save the model
            
        Returns:
        --------
        MLStockPredictor
            Trained ML model
        """
        # Create model directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Create model path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{model_type}_{task_type}_{timestamp}.joblib")
        
        # Initialize ML model
        ml_model = MLStockPredictor(
            model_type=model_type,
            task_type=task_type,
            model_path=model_path if save_model else None
        )
        
        # Prepare data
        data = ml_model.prepare_data(df, target_column, test_size=test_size)
        if data is None:
            return None
        
        X_train, X_test, y_train, y_test = data
        
        # Train model
        ml_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = ml_model.evaluate(X_test, y_test)
        
        # Store model
        model_name = f"{model_type}_{task_type}_{target_column}"
        self.models[model_name] = ml_model
        
        logger.info(f"Trained {model_type} model for {target_column} prediction")
        
        return ml_model
    
    def train_multiple_models(self, df, price_target='Close', direction_target='Target_Direction_1d',
                             save_models=True):
        """
        Train multiple models for comparison
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with preprocessed data
        price_target : str
            Column to predict for regression models
        direction_target : str
            Column to predict for classification models
        save_models : bool
            Whether to save the models
            
        Returns:
        --------
        dict
            Dictionary with trained models
        """
        # Train LSTM model for price prediction
        lstm_model = self.train_lstm_model(
            df, 
            target_column=price_target,
            save_model=save_models
        )
        
        # Train random forest for price prediction
        rf_regression = self.train_ml_model(
            df,
            target_column=price_target,
            model_type='random_forest',
            task_type='regression',
            save_model=save_models
        )
        
        # Train XGBoost for price prediction
        xgb_regression = self.train_ml_model(
            df,
            target_column=price_target,
            model_type='xgboost',
            task_type='regression',
            save_model=save_models
        )
        
        # Train random forest for direction prediction
        rf_classification = self.train_ml_model(
            df,
            target_column=direction_target,
            model_type='random_forest',
            task_type='classification',
            save_model=save_models
        )
        
        # Train XGBoost for direction prediction
        xgb_classification = self.train_ml_model(
            df,
            target_column=direction_target,
            model_type='xgboost',
            task_type='classification',
            save_model=save_models
        )
        
        # Train SVM for direction prediction
        svm_classification = self.train_ml_model(
            df,
            target_column=direction_target,
            model_type='svm',
            task_type='classification',
            save_model=save_models
        )
        
        return self.models
    
    def predict_price(self, df, model_name=None, steps_ahead=5):
        """
        Predict stock prices
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        model_name : str, optional
            Name of model to use for prediction
        steps_ahead : int
            Number of steps ahead to predict
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with predictions
        """
        # Handle model selection
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            # Use the first LSTM model if available
            lstm_models = {name: model for name, model in self.models.items() 
                          if isinstance(model, LSTMStockPredictor)}
            
            if lstm_models:
                model_name = list(lstm_models.keys())[0]
                model = lstm_models[model_name]
            else:
                # If no LSTM model, try to use a regression ML model
                regression_models = {name: model for name, model in self.models.items() 
                                   if isinstance(model, MLStockPredictor) and model.task_type == 'regression'}
                
                if regression_models:
                    model_name = list(regression_models.keys())[0]
                    model = regression_models[model_name]
                else:
                    logger.error("No suitable prediction model available")
                    return None
        
        # Make predictions
        if isinstance(model, LSTMStockPredictor):
            predictions = model.predict(df, steps_ahead=steps_ahead)
        elif isinstance(model, MLStockPredictor) and model.task_type == 'regression':
            # For ML models, implement multi-step prediction by iteratively predicting one step at a time
            # Start with the most recent data
            X_latest = df.iloc[-1:].copy()
            
            # Initialize list to store predictions
            multi_step_preds = []
            dates = []
            
            # Get the last date from the input data
            last_date = df.index[-1]
            
            # Iterate for each step
            for i in range(steps_ahead):
                # Make one-step prediction
                pred = model.predict(X_latest)
                multi_step_preds.append(pred[0])
                
                # Create next date
                next_date = last_date + pd.Timedelta(days=i+1)
                # Skip weekends
                while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                    next_date = next_date + pd.Timedelta(days=1)
                dates.append(next_date)
                
                # Create new row for next prediction by copying the last row
                next_row = X_latest.iloc[-1:].copy()
                
                # Update the target column with the prediction
                target_columns = ['Close', 'Open', 'High', 'Low']
                for col in next_row.columns:
                    if col in target_columns:
                        if col == 'Close':
                            next_row[col] = pred[0]
                        elif col == 'Open':
                            next_row[col] = pred[0] * 0.995  # Slight offset for Open
                        elif col == 'High':
                            next_row[col] = pred[0] * 1.01   # Slight increase for High
                        elif col == 'Low':
                            next_row[col] = pred[0] * 0.99   # Slight decrease for Low
                
                # Create a new dataframe with the next row and append to X_latest
                next_df = pd.DataFrame(next_row.values, index=[next_date], columns=next_row.columns)
                X_latest = pd.concat([X_latest, next_df])
            
            # Create a DataFrame with the predictions
            predictions_df = pd.DataFrame(multi_step_preds, index=dates, columns=['Close'])
            predictions = predictions_df
        else:
            logger.error(f"Model {model_name} is not suitable for price prediction")
            return None
        
        logger.info(f"Made {steps_ahead} step-ahead predictions using {model_name}")
        
        return predictions
    
    def predict_next_n_days(self, df, model_name=None, n_days=5):
        """
        Predict stock prices for the next n days
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        model_name : str, optional
            Name of model to use for prediction
        n_days : int
            Number of days to predict ahead
            
        Returns:
        --------
        list
            List of predicted prices for the next n days
        """
        # This is a simpler interface for the predict_price method that returns
        # just an array of values rather than a DataFrame
        
        predictions_df = self.predict_price(df, model_name=model_name, steps_ahead=n_days)
        
        if predictions_df is None:
            return None
            
        # Extract the predicted values into a simple list
        if isinstance(predictions_df, pd.DataFrame):
            # If it's a DataFrame, extract the 'Close' column or the first column
            if 'Close' in predictions_df.columns:
                predictions = predictions_df['Close'].values.tolist()
            else:
                predictions = predictions_df.iloc[:, 0].values.tolist()
        else:
            # If it's not a DataFrame, convert to list if it's a numpy array
            predictions = predictions_df.tolist() if hasattr(predictions_df, 'tolist') else list(predictions_df)
            
        return predictions[:n_days]  # Ensure we return exactly n_days predictions
    
    def predict_direction(self, df, model_name=None):
        """
        Predict stock price direction
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        model_name : str, optional
            Name of model to use for prediction
            
        Returns:
        --------
        numpy.ndarray
            Array with direction predictions (0 = down, 1 = up)
        """
        # Handle model selection
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            # Use the first classification model if available
            classification_models = {name: model for name, model in self.models.items() 
                                    if isinstance(model, MLStockPredictor) and model.task_type == 'classification'}
            
            if classification_models:
                model_name = list(classification_models.keys())[0]
                model = classification_models[model_name]
            else:
                logger.error("No classification model available for prediction")
                return None
        
        # Make predictions
        if isinstance(model, MLStockPredictor) and model.task_type == 'classification':
            predictions = model.predict(df)
        else:
            logger.error(f"Model {model_name} is not suitable for direction prediction")
            return None
        
        logger.info(f"Made direction predictions using {model_name}")
        
        return predictions
    
    def evaluate_models(self, test_df, target_column='Close', save_results=True):
        """
        Evaluate all trained models
        
        Parameters:
        -----------
        test_df : pandas.DataFrame
            DataFrame with test data
        target_column : str
            Column to evaluate
        save_results : bool
            Whether to save evaluation results
            
        Returns:
        --------
        dict
            Dictionary with evaluation results
        """
        results = {}
        
        # Group models by type
        regression_models = {name: model for name, model in self.models.items() 
                           if isinstance(model, MLStockPredictor) and model.task_type == 'regression'
                           or isinstance(model, LSTMStockPredictor)}
        
        classification_models = {name: model for name, model in self.models.items() 
                               if isinstance(model, MLStockPredictor) and model.task_type == 'classification'}
        
        # Evaluate regression models
        if regression_models:
            # Prepare data for comparison
            y_true = test_df[target_column].values
            regression_data = []
            
            for name, model in regression_models.items():
                if isinstance(model, LSTMStockPredictor):
                    # LSTM predictions
                    pred_df = model.predict(test_df, use_latest_window=False)
                    y_pred = pred_df[target_column].values
                else:
                    # ML predictions
                    y_pred = model.predict(test_df)
                
                # Make sure predictions and true values have the same length
                min_len = min(len(y_true), len(y_pred))
                y_true_trim = y_true[:min_len]
                y_pred_trim = y_pred[:min_len]
                
                regression_data.append((y_true_trim, y_pred_trim, name))
                
                logger.info(f"Model {name} prediction array size: {len(y_pred_trim)}, true values size: {len(y_true_trim)}")
            
            # Compare regression models
            regression_results = self.evaluator.compare_regression_models(
                regression_data, save_results=save_results
            )
            
            results['regression'] = regression_results
            
            # Plot predictions comparison
            y_preds = [data[1] for data in regression_data]
            model_names = [data[2] for data in regression_data]
            
            self.evaluator.plot_predictions(
                y_true_trim, y_preds, model_names, 
                title=f'{target_column} Prediction Comparison',
                save_plot=save_results
            )
        
        # Evaluate classification models
        if classification_models:
            # Get direction target
            direction_cols = [col for col in test_df.columns if col.startswith('Target_Direction')]
            if direction_cols:
                direction_target = direction_cols[0]
                
                # Prepare data for comparison
                y_true = test_df[direction_target].values
                classification_data = []
                
                for name, model in classification_models.items():
                    y_pred = model.predict(test_df)
                    
                    # Make sure predictions and true values have the same length
                    min_len = min(len(y_true), len(y_pred))
                    y_true_trim = y_true[:min_len]
                    y_pred_trim = y_pred[:min_len]
                    
                    try:
                        y_proba = model.predict_proba(test_df)
                        # Also trim probability array if needed
                        if len(y_proba) > min_len:
                            y_proba_trim = y_proba[:min_len]
                        else:
                            y_proba_trim = y_proba
                        classification_data.append((y_true_trim, y_pred_trim, y_proba_trim, name))
                    except:
                        classification_data.append((y_true_trim, y_pred_trim, name))
                    
                    logger.info(f"Model {name} prediction array size: {len(y_pred_trim)}, true values size: {len(y_true_trim)}")
                
                # Compare classification models
                classification_results = self.evaluator.compare_classification_models(
                    classification_data, save_results=save_results
                )
                
                results['classification'] = classification_results
        
        # Plot feature importance for random forest models
        for name, model in self.models.items():
            if isinstance(model, MLStockPredictor) and hasattr(model.model, 'feature_importances_'):
                feature_importances = pd.Series(model.model.feature_importances_, index=model.feature_names)
                
                self.evaluator.plot_feature_importance(
                    feature_importances,
                    title=f'Feature Importance - {name}',
                    save_plot=save_results
                )
        
        return results
    
    def save_models(self, path=None):
        """
        Save all models
        
        Parameters:
        -----------
        path : str, optional
            Custom path to save models
        """
        for name, model in self.models.items():
            if path:
                model_path = os.path.join(path, f"{name}.model")
            else:
                model_path = None
            
            model.save(model_path)
    
    def load_model(self, model_path, model_type='lstm'):
        """
        Load a saved model
        
        Parameters:
        -----------
        model_path : str
            Path to model file
        model_type : str
            Type of model ('lstm' or 'ml')
            
        Returns:
        --------
        object
            Loaded model
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return None
        
        try:
            if model_type.lower() == 'lstm':
                model = LSTMStockPredictor(model_path=model_path)
                model.load()
            else:
                model = MLStockPredictor(model_path=model_path)
                model.load()
            
            # Extract model name from path
            model_name = os.path.basename(model_path).split('.')[0]
            self.models[model_name] = model
            
            logger.info(f"Loaded model from {model_path}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def run_forecasting_pipeline(self, ticker, train_test_split=0.8, prediction_days=5, 
                                save_models=True, save_results=True):
        """
        Run complete forecasting pipeline
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        train_test_split : float
            Proportion of data to use for training
        prediction_days : int
            Number of days to predict ahead
        save_models : bool
            Whether to save trained models
        save_results : bool
            Whether to save evaluation results
            
        Returns:
        --------
        dict
            Dictionary with pipeline results
        """
        # Load data
        df = self.load_stock_data(ticker)
        if df is None:
            return None
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Split into train and test sets
        train_size = int(len(processed_df) * train_test_split)
        train_df = processed_df.iloc[:train_size]
        test_df = processed_df.iloc[train_size:]
        
        # Train multiple models
        self.train_multiple_models(
            train_df,
            price_target='Close',
            direction_target='Target_Direction_1d',
            save_models=save_models
        )
        
        # Evaluate models
        evaluation_results = self.evaluate_models(
            test_df,
            target_column='Close',
            save_results=save_results
        )
        
        # Make future predictions
        latest_data = processed_df.iloc[-100:]  # Use last 100 data points for prediction
        
        price_predictions = self.predict_price(
            latest_data,
            steps_ahead=prediction_days
        )
        
        direction_predictions = self.predict_direction(latest_data)
        
        # Combine results
        results = {
            'ticker': ticker,
            'data_shape': processed_df.shape,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'models': list(self.models.keys()),
            'evaluation': evaluation_results,
            'price_predictions': price_predictions,
            'direction_predictions': direction_predictions
        }
        
        # Save final results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.results_dir, f"{ticker}_forecast_results_{timestamp}.pkl")
            
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Saved forecast results to {results_path}")
        
        return results