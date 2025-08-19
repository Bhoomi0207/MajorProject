"""
Traditional ML models for stock price prediction.
"""
import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class MLStockPredictor:
    """
    Machine learning models for stock price prediction
    """
    def __init__(self, model_type='random_forest', task_type='regression', model_path=None):
        """
        Initialize ML model

        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'xgboost', 'svm', 'linear')
        task_type : str
            Type of task ('regression' or 'classification')
        model_path : str
            Path to save/load model
        """
        self.model_type = model_type.lower()
        self.task_type = task_type.lower()
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Create model directory if it doesn't exist
        if model_path and not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

    def _get_model(self):
        """
        Get model instance based on model_type and task_type
        
        Returns:
        --------
        object
            Model instance
        """
        if self.task_type == 'regression':
            if self.model_type == 'random_forest':
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'xgboost':
                return xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            elif self.model_type == 'svm':
                return SVR(kernel='rbf')
            elif self.model_type == 'linear':
                return LinearRegression()
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return None
        elif self.task_type == 'classification':
            if self.model_type == 'random_forest':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'xgboost':
                return xgb.XGBClassifier(random_state=42)
            elif self.model_type == 'svm':
                return SVC(kernel='rbf', probability=True)
            elif self.model_type == 'linear':
                return LogisticRegression(random_state=42)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return None
        else:
            logger.error(f"Unknown task type: {self.task_type}")
            return None
        
    def prepare_data(self, df, target_column, test_size=0.2, use_only_numeric=True):
        """
        Prepare data for ML model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        target_column : str
            Column to predict
        test_size : float
            Proportion of data to use for testing
        use_only_numeric : bool
            Whether to use only numeric columns
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if target_column not in df.columns:
            logger.error(f"Target column {target_column} not in DataFrame")
            return None
        
        # Split data
        train_size = int(len(df) * (1 - test_size))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Select features
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        
        # Use only numeric columns if specified
        if use_only_numeric:
            X_train = X_train.select_dtypes(include=['number'])
            X_test = X_test.select_dtypes(include=['number'])
        
        # Store column names
        self.feature_names = X_train.columns
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        logger.info(f"Prepared data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
        logger.info(f"Using {len(self.feature_names)} features")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train):
        """
        Train ML model
        
        Parameters:
        -----------
        X_train : pandas.DataFrame
            Training features
        y_train : pandas.Series
            Training target
            
        Returns:
        --------
        object
            Trained model
        """
        # Get model
        if self.model is None:
            self.model = self._get_model()
        
        if self.model is None:
            return None
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        logger.info(f"Model training complete")
        
        return self.model
    
    def predict(self, X, steps_ahead=1):
        """
        Make predictions with ML model
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features
        steps_ahead : int, optional
            Number of steps ahead to predict (used for compatibility with LSTM interface)
            
        Returns:
        --------
        numpy.ndarray or pandas.DataFrame
            Predictions
        """
        if not self.is_fitted or self.model is None:
            logger.error("Model not fitted yet")
            return None
        
        # Scale features if needed
        if isinstance(X, pd.DataFrame):
            # Check columns
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                logger.error(f"Missing columns in input data: {missing_cols}")
                return None
            
            # Select and order columns
            X = X[self.feature_names]
            
            # Scale features
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names, index=X.index)
        
        # Make predictions for current data
        predictions = self.model.predict(X)
        
        # If steps_ahead > 1 and this is a regression model, we need to simulate future predictions
        if steps_ahead > 1 and self.task_type == 'regression':
            # We'll create a DataFrame to return predictions with dates
            if isinstance(X, pd.DataFrame) and len(X.index) > 0:
                last_date = X.index[-1]
                dates = pd.date_range(start=last_date, periods=steps_ahead+1)[1:]
                
                # For ML models, we'll just use the last prediction and add some small random variation
                # This is a simple approach - in a real system you'd want a more sophisticated method
                last_pred = predictions[-1]
                future_preds = [last_pred]
                
                for i in range(1, steps_ahead):
                    # Add small random variation (±2%)
                    variation = np.random.uniform(-0.02, 0.02)
                    next_pred = future_preds[-1] * (1 + variation)
                    future_preds.append(next_pred)
                
                # Create a DataFrame with the future predictions
                pred_df = pd.DataFrame({'Close': future_preds}, index=dates)
                return pred_df
            
            # If we don't have dates, just return the array
            return predictions
        
        return predictions
    
    def predict_proba(self, X):
        """
        Make probability predictions with ML model (for classification only)
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features
            
        Returns:
        --------
        numpy.ndarray
            Probability predictions
        """
        if not self.is_fitted or self.model is None:
            logger.error("Model not fitted yet")
            return None
        
        if self.task_type != 'classification':
            logger.error("predict_proba() is only available for classification models")
            return None
        
        # Scale features if needed
        if isinstance(X, pd.DataFrame):
            # Check columns
            missing_cols = set(self.feature_names) - set(X.columns)
            if missing_cols:
                logger.error(f"Missing columns in input data: {missing_cols}")
                return None
            
            # Select and order columns
            X = X[self.feature_names]
            
            # Scale features
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names, index=X.index)
        
        # Make predictions
        try:
            predictions = self.model.predict_proba(X)
            return predictions
        except AttributeError:
            logger.error(f"Model {self.model_type} does not support predict_proba()")
            return None
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test target
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if not self.is_fitted or self.model is None:
            logger.error("Model not fitted yet")
            return None
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
            logger.info(f"Model evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
            top_features = feature_importances.sort_values(ascending=False).head(10)
            logger.info(f"Top 10 important features:\n{top_features}")
            metrics['feature_importances'] = feature_importances
        
        return metrics
    
    def save(self, path=None):
        """
        Save model to file
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        if not self.is_fitted or self.model is None:
            logger.error("Model not fitted yet")
            return
        
        save_path = path or self.model_path
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"models/{self.model_type}_{self.task_type}_{timestamp}.joblib"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'task_type': self.task_type
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path=None):
        """
        Load model from file
        
        Parameters:
        -----------
        path : str
            Path to load model from
        """
        load_path = path or self.model_path
        if not load_path:
            logger.error("No model path specified")
            return
        
        if not os.path.exists(load_path):
            logger.error(f"Model file {load_path} not found")
            return
        
        # Load model
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.task_type = model_data['task_type']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {load_path}")