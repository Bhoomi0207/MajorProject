"""
LSTM-based deep learning model for stock price prediction.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    """
    LSTM model for predicting stock prices
    """
    def __init__(self, sequence_length=10, validation_split=0.2, model_path=None):
        """
        Initialize LSTM model

        Parameters:
        -----------
        sequence_length : int
            Number of time steps to look back
        validation_split : float
            Proportion of training data to use for validation
        model_path : str
            Path to save/load model
        """
        self.sequence_length = sequence_length
        self.validation_split = validation_split
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        
        # Default target
        self.target_column = 'Close'
        
        # Create model directory if it doesn't exist
        if model_path and not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
    
    def _create_sequences(self, data):
        """
        Create sequences for LSTM input
        
        Parameters:
        -----------
        data : numpy.ndarray
            Feature data
            
        Returns:
        --------
        tuple
            (X, y) sequences
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, output_shape=1):
        """
        Build LSTM model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, n_features)
        output_shape : int
            Number of output values to predict
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Built model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=output_shape))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logger.info(f"Built LSTM model with input shape {input_shape} and output shape {output_shape}")
        return model
    
    def prepare_data(self, df, target_column=None):
        """
        Prepare data for LSTM model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        target_column : str
            Column to predict (default: Close)
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_val, y_val)
        """
        if target_column:
            self.target_column = target_column
        
        if self.target_column not in df.columns:
            logger.error(f"Target column {self.target_column} not in DataFrame")
            return None
        
        # Select features and target
        features = df.copy()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Split into train and validation sets
        train_size = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        logger.info(f"Prepared data with {X_train.shape[0]} training sequences and {X_val.shape[0]} validation sequences")
        
        return X_train, y_train, X_val, y_val
    
    def fit(self, df, target_column=None, epochs=100, batch_size=32, patience=10):
        """
        Train LSTM model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        target_column : str
            Column to predict (default: Close)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Patience for early stopping
            
        Returns:
        --------
        tensorflow.keras.callbacks.History
            Training history
        """
        # Prepare data
        data = self.prepare_data(df, target_column)
        if data is None:
            return None
        
        X_train, y_train, X_val, y_val = data
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(early_stopping)
        
        # Model checkpoint
        if self.model_path:
            checkpoint = ModelCheckpoint(
                self.model_path, 
                monitor='val_loss', 
                save_best_only=True
            )
            callbacks.append(checkpoint)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        logger.info(f"Model training complete: {len(history.epoch)} epochs, final loss: {history.history['loss'][-1]:.4f}, final val_loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
    def predict(self, df, steps_ahead=1, use_latest_window=True):
        """
        Make predictions with LSTM model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        steps_ahead : int
            Number of steps ahead to predict
        use_latest_window : bool
            Whether to use only the latest available window for prediction
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with predictions
        """
        if not self.is_fitted or self.model is None:
            logger.error("Model not fitted yet")
            return None
        
        # Scale data
        scaled_data = self.scaler.transform(df)
        n_features = scaled_data.shape[1]
        
        if use_latest_window:
            # Use only the latest window for prediction
            input_data = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, n_features)
            
            # Make predictions for multiple steps ahead
            predictions = []
            current_input = input_data.copy()
            
            for _ in range(steps_ahead):
                # Predict next step
                pred = self.model.predict(current_input)
                predictions.append(pred[0])
                
                # Update input for next prediction (roll window forward)
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, :] = pred[0]
            
            # Convert predictions to proper format for inverse_transform
            predictions = np.array(predictions)
            
            # Debug information
            logger.info(f"Predictions shape before reshaping: {predictions.shape}")
            logger.info(f"Number of features in original data: {n_features}")
            
            # Handle single feature prediction case (when output is just a single value per step)
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                # Create a properly shaped array for inverse_transform with all zeros
                full_sequence = np.zeros((len(predictions), n_features))
                
                # The target column index (assuming it's the first feature for simplicity)
                target_idx = 0
                if self.target_column in df.columns:
                    target_idx = df.columns.get_loc(self.target_column)
                    
                # Copy the predicted values to the target column
                full_sequence[:, target_idx] = predictions[:, 0]
            else:
                # If prediction has same number of features, use as is
                full_sequence = predictions
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(full_sequence)
            
            # If we only care about certain columns, extract them
            if self.target_column in df.columns:
                target_idx = df.columns.get_loc(self.target_column)
                predictions_target = predictions[:, target_idx].reshape(-1, 1)
                
                # Create DataFrame with just target column prediction
                dates = pd.date_range(start=df.index[-1], periods=steps_ahead+1)[1:]
                pred_df = pd.DataFrame(predictions_target, index=dates, columns=[self.target_column])
            else:
                # Create DataFrame with all predictions
                dates = pd.date_range(start=df.index[-1], periods=steps_ahead+1)[1:]
                pred_df = pd.DataFrame(predictions, index=dates, columns=df.columns)
            
            return pred_df
        
        else:
            # Create sequences for all data
            X, _ = self._create_sequences(scaled_data)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Debug information
            logger.info(f"Predictions shape for historical data: {predictions.shape}")
            
            # Handle single feature prediction case
            if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                # Create a properly shaped array for inverse_transform with all zeros
                full_sequence = np.zeros((len(predictions), n_features))
                
                # The target column index
                target_idx = df.columns.get_loc(self.target_column) if self.target_column in df.columns else 0
                
                # Copy the predicted values to the target column
                full_sequence[:, target_idx] = predictions[:, 0]
            else:
                # If prediction has same number of features, use as is
                full_sequence = predictions
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(full_sequence)
            
            # Extract target column if specified
            if self.target_column in df.columns:
                target_idx = df.columns.get_loc(self.target_column)
                predictions_target = predictions[:, target_idx].reshape(-1, 1)
                
                # Create DataFrame with just target column prediction
                dates = df.index[self.sequence_length:]
                pred_df = pd.DataFrame(predictions_target, index=dates, columns=[self.target_column])
            else:
                # Create DataFrame with all predictions
                dates = df.index[self.sequence_length:]
                pred_df = pd.DataFrame(predictions, index=dates, columns=df.columns)
            
            return pred_df
    
    def evaluate(self, df, target_column=None):
        """
        Evaluate model on test data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        target_column : str
            Column to evaluate (default: Close)
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        if not self.is_fitted or self.model is None:
            logger.error("Model not fitted yet")
            return None
        
        if target_column:
            self.target_column = target_column
        
        # Prepare data
        data = self.prepare_data(df, self.target_column)
        if data is None:
            return None
        
        _, _, X_val, y_val = data
        
        # Evaluate model
        loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        
        # Prepare full sequences for inverse transform
        y_val_full = np.zeros((len(y_val), y_val.shape[1]))
        y_val_full[:, :] = y_val
        
        y_pred_full = np.zeros((len(y_pred), y_pred.shape[1]))
        y_pred_full[:, :] = y_pred
        
        # Inverse transform
        y_val_inv = self.scaler.inverse_transform(y_val_full)
        y_pred_inv = self.scaler.inverse_transform(y_pred_full)
        
        # Calculate metrics
        mse = np.mean(np.square(y_val_inv - y_pred_inv))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_val_inv - y_pred_inv))
        
        metrics = {
            'loss': loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Model evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
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
            save_path = f"models/lstm_model_{timestamp}.h5"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        self.model.save(save_path)
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
        self.model = tf.keras.models.load_model(load_path)
        self.is_fitted = True
        logger.info(f"Model loaded from {load_path}")