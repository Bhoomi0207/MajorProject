"""
Framework for evaluating and comparing stock price prediction models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import os
from datetime import datetime

# Local imports
from src.predictive_models.lstm_model import LSTMStockPredictor
from src.predictive_models.ml_models import MLStockPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating and comparing stock price prediction models
    """
    def __init__(self, output_dir='results'):
        """
        Initialize model evaluator
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def evaluate_regression_model(self, y_true, y_pred, model_name):
        """
        Evaluate regression model performance
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of model
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'model': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Model {model_name} evaluation - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def evaluate_classification_model(self, y_true, y_pred, y_proba=None, model_name='', labels=None):
        """
        Evaluate classification model performance
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        y_proba : array-like, optional
            Predicted probabilities
        model_name : str
            Name of model
        labels : list, optional
            Class labels
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"Model {model_name} evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def compare_regression_models(self, models_data, save_results=True):
        """
        Compare multiple regression models
        
        Parameters:
        -----------
        models_data : list of tuples
            List of (y_true, y_pred, model_name) tuples
        save_results : bool
            Whether to save comparison results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with comparison results
        """
        results = []
        
        for y_true, y_pred, model_name in models_data:
            metrics = self.evaluate_regression_model(y_true, y_pred, model_name)
            results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by RMSE
        comparison_df = comparison_df.sort_values('rmse')
        
        # Save results if needed
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"regression_comparison_{timestamp}.csv")
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Comparison results saved to {filepath}")
            
            # Create comparison chart
            self._plot_regression_comparison(comparison_df, timestamp)
        
        return comparison_df
    
    def compare_classification_models(self, models_data, save_results=True):
        """
        Compare multiple classification models
        
        Parameters:
        -----------
        models_data : list of tuples
            List of (y_true, y_pred, model_name) tuples
        save_results : bool
            Whether to save comparison results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with comparison results
        """
        results = []
        
        for data in models_data:
            if len(data) == 3:
                y_true, y_pred, model_name = data
                y_proba = None
            else:
                y_true, y_pred, y_proba, model_name = data
            
            metrics = self.evaluate_classification_model(y_true, y_pred, y_proba, model_name)
            results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by F1 score
        comparison_df = comparison_df.sort_values('f1', ascending=False)
        
        # Save results if needed
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"classification_comparison_{timestamp}.csv")
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Comparison results saved to {filepath}")
            
            # Create comparison chart
            self._plot_classification_comparison(comparison_df, timestamp)
        
        return comparison_df
    
    def plot_predictions(self, y_true, y_preds, model_names, title='Model Predictions Comparison', save_plot=True):
        """
        Plot model predictions against true values
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_preds : list of array-like
            List of predictions from different models
        model_names : list of str
            Names of models
        title : str
            Plot title
        save_plot : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        plt.figure(figsize=(15, 8))
        
        # Plot true values
        plt.plot(y_true, label='True Values', linewidth=2)
        
        # Plot predictions for each model
        for y_pred, model_name in zip(y_preds, model_names):
            plt.plot(y_pred, label=f'{model_name} Predictions', linewidth=1.5, alpha=0.8)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save plot if needed
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"predictions_comparison_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved to {filepath}")
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_importances, top_n=20, title='Feature Importance', save_plot=True):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_importances : pandas.Series
            Feature importance values with feature names as index
        top_n : int
            Number of top features to show
        title : str
            Plot title
        save_plot : bool
            Whether to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Sort and select top features
        top_features = feature_importances.sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_features.values, y=top_features.index)
        plt.title(title, fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save plot if needed
        if save_plot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.output_dir, f"feature_importance_{timestamp}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {filepath}")
        
        return plt.gcf()
    
    def _plot_regression_comparison(self, comparison_df, timestamp):
        """
        Plot comparison of regression models
        
        Parameters:
        -----------
        comparison_df : pandas.DataFrame
            DataFrame with comparison results
        timestamp : str
            Timestamp for filename
        """
        # Create bar chart of RMSE and MAE
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, comparison_df['rmse'], width, label='RMSE')
        ax.bar(x + width/2, comparison_df['mae'], width, label='MAE')
        
        ax.set_title('Regression Models Comparison', fontsize=16)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, f"regression_comparison_plot_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Regression comparison plot saved to {filepath}")
    
    def _plot_classification_comparison(self, comparison_df, timestamp):
        """
        Plot comparison of classification models
        
        Parameters:
        -----------
        comparison_df : pandas.DataFrame
            DataFrame with comparison results
        timestamp : str
            Timestamp for filename
        """
        # Create bar chart of accuracy, precision, recall, and F1
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(comparison_df))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1.5)
            ax.bar(x + offset, comparison_df[metric], width, label=metric.capitalize())
        
        ax.set_title('Classification Models Comparison', fontsize=16)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, f"classification_comparison_plot_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Classification comparison plot saved to {filepath}")

    def evaluate_prediction_accuracy(self, predictions_df, actual_df, target_column='Close', plot=True, save_plot=False):
        """
        Compare predicted values with actual values to measure prediction accuracy
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame with predicted values
        actual_df : pandas.DataFrame
            DataFrame with actual values
        target_column : str
            Column to compare
        plot : bool
            Whether to create a plot
        save_plot : bool
            Whether to save the plot
            
        Returns:
        --------
        dict
            Dictionary with accuracy metrics
        """
        if target_column not in predictions_df.columns or target_column not in actual_df.columns:
            logging.error(f"Target column {target_column} not in both dataframes")
            return None
        
        # Align dates between prediction and actual
        # Convert index to datetime if needed
        predictions_df = predictions_df.copy()
        actual_df = actual_df.copy()
        
        if not isinstance(predictions_df.index, pd.DatetimeIndex):
            predictions_df.index = pd.to_datetime(predictions_df.index)
        
        if not isinstance(actual_df.index, pd.DatetimeIndex):
            actual_df.index = pd.to_datetime(actual_df.index)
        
        # Get dates that are in both dataframes
        common_dates = predictions_df.index.intersection(actual_df.index)
        
        if len(common_dates) == 0:
            logging.warning("No overlapping dates between prediction and actual data")
            # Try to see if prediction dates are future dates compared to training data
            return {
                'mse': None,
                'rmse': None,
                'mae': None,
                'mape': None,
                'accuracy_pct': None,
                'common_dates': 0,
                'direction_accuracy': None
            }
        
        # Filter to only common dates
        pred_values = predictions_df.loc[common_dates, target_column].values
        actual_values = actual_df.loc[common_dates, target_column].values
        
        # Calculate metrics
        mse = np.mean((pred_values - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_values - actual_values))
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100
        
        # Calculate accuracy percentage (100 - MAPE)
        accuracy_pct = max(0, 100 - mape)  # Cap at 0 if MAPE exceeds 100%
        
        # Calculate direction accuracy (up/down)
        actual_direction = np.diff(actual_values) > 0
        pred_direction = np.diff(pred_values) > 0
        
        if len(actual_direction) > 0:
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = None
        
        # Create plot
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(common_dates, actual_values, label='Actual', marker='o')
            plt.plot(common_dates, pred_values, label='Predicted', marker='x')
            
            plt.title(f'Prediction Accuracy: {accuracy_pct:.2f}% ({len(common_dates)} days)')
            plt.xlabel('Date')
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add a text box with metrics
            textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nAccuracy: {accuracy_pct:.2f}%'
            if direction_accuracy is not None:
                textstr += f'\nDirection Accuracy: {direction_accuracy:.2f}%'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            if save_plot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f'prediction_accuracy_{timestamp}.png')
                plt.savefig(output_path)
                logging.info(f"Saved prediction accuracy plot to {output_path}")
            
            plt.tight_layout()
            
            # Return the figure instead of showing it
            fig = plt.gcf()
            plt.close()
        else:
            fig = None
        
        # Return metrics
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'accuracy_pct': accuracy_pct,
            'common_dates': len(common_dates),
            'direction_accuracy': direction_accuracy,
            'figure': fig
        }

    def backtest_predictions(self, df, model, prediction_days, target_column='Close', 
                            backtest_periods=5, test_size=0.2, save_results=False):
        """
        Backtest model predictions against historical data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Full historical DataFrame
        model : object
            Prediction model (must have 'predict' method)
        prediction_days : int
            Number of days to predict ahead
        target_column : str
            Column to predict
        backtest_periods : int
            Number of backtest periods to run
        test_size : float
            Size of test period as a fraction of data
        save_results : bool
            Whether to save the backtest results
            
        Returns:
        --------
        dict
            Dictionary with backtest results
        """
        logging.info(f"Starting backtest with {backtest_periods} periods, {prediction_days} days prediction horizon")
        
        results = []
        figures = []
        
        # Get total size to test
        data_size = len(df)
        test_size_days = max(int(data_size * test_size), prediction_days * 2)
        
        # Calculate stride (how far to move for each backtest period)
        if backtest_periods > 1:
            stride = test_size_days // backtest_periods
        else:
            stride = 0
        
        # Terminal output header for results
        print("\n" + "="*80)
        print(f"MODEL ACCURACY EVALUATION - BACKTEST RESULTS ({prediction_days}-day predictions)")
        print("="*80)
        print(f"Model Type: {model.__class__.__name__}")
        print(f"Target Column: {target_column}")
        print(f"Backtest Periods: {backtest_periods}")
        print(f"Test Size: {test_size_days} days per period")
        print("-"*80)
        print(f"{'Period':<8} {'Price Acc.%':<12} {'Direction Acc.%':<16} {'RMSE':<10} {'MAE':<10} {'Date Range'}")
        print("-"*80)
        
        for i in range(backtest_periods):
            # Calculate test start/end indices
            test_end = data_size - (i * stride)
            test_start = test_end - test_size_days
            
            if test_start < 0:
                logging.warning(f"Not enough data for backtest period {i+1}")
                break
            
            # Split into train/test
            train_df = df.iloc[:test_start].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            logging.info(f"Backtest period {i+1}: train_size={len(train_df)}, test_size={len(test_df)}")
            
            try:
                # Generate prediction
                prediction_df = model.predict(train_df, steps_ahead=prediction_days)
                
                if prediction_df is None or prediction_df.empty:
                    logging.error(f"No predictions generated for backtest period {i+1}")
                    continue
                
                # Evaluate against actual data
                evaluation = self.evaluate_prediction_accuracy(
                    prediction_df, 
                    test_df, 
                    target_column=target_column,
                    plot=(i < 3),  # Only create plots for the first 3 periods to save memory
                    save_plot=save_results
                )
                
                if evaluation:
                    # Add period info to results
                    evaluation['period'] = i + 1
                    evaluation['train_size'] = len(train_df)
                    evaluation['test_size'] = len(test_df)
                    evaluation['train_end_date'] = train_df.index[-1]
                    evaluation['test_start_date'] = test_df.index[0]
                    evaluation['test_end_date'] = test_df.index[-1]
                    
                    # Terminal output for this period
                    date_range = f"{test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}"
                    acc_pct = evaluation.get('accuracy_pct', 0)
                    dir_acc = evaluation.get('direction_accuracy', 0)
                    rmse = evaluation.get('rmse', 0)
                    mae = evaluation.get('mae', 0)
                    
                    print(f"{i+1:<8} {acc_pct:<12.2f} {dir_acc if dir_acc else 'N/A':<16} {rmse:<10.4f} {mae:<10.4f} {date_range}")
                    
                    # Save figure if created
                    if 'figure' in evaluation and evaluation['figure'] is not None:
                        figures.append(evaluation['figure'])
                        # Remove figure from results to avoid serialization issues
                        evaluation_results = {k: v for k, v in evaluation.items() if k != 'figure'}
                        results.append(evaluation_results)
                    else:
                        results.append(evaluation)
                
            except Exception as e:
                logging.error(f"Error in backtest period {i+1}: {e}")
                print(f"Error in backtest period {i+1}: {e}")
                continue
        
        # Calculate aggregate metrics
        if results:
            # Average metrics across all periods
            aggregate_metrics = {
                'avg_rmse': np.mean([r['rmse'] for r in results if r['rmse'] is not None]),
                'avg_mae': np.mean([r['mae'] for r in results if r['mae'] is not None]),
                'avg_mape': np.mean([r['mape'] for r in results if r['mape'] is not None]),
                'avg_accuracy': np.mean([r['accuracy_pct'] for r in results if r['accuracy_pct'] is not None]),
                'avg_direction_accuracy': np.mean([r['direction_accuracy'] for r in results if r['direction_accuracy'] is not None]),
                'total_periods': len(results),
                'prediction_days': prediction_days
            }
            
            # Terminal output for aggregate results
            print("-"*80)
            print("AGGREGATE RESULTS:")
            print(f"Average Price Accuracy:    {aggregate_metrics['avg_accuracy']:.2f}%")
            print(f"Average Direction Accuracy:{aggregate_metrics['avg_direction_accuracy']:.2f}%")
            print(f"Average RMSE:              {aggregate_metrics['avg_rmse']:.4f}")
            print(f"Average MAE:               {aggregate_metrics['avg_mae']:.4f}")
            print(f"Average MAPE:              {aggregate_metrics['avg_mape']:.2f}%")
            print("="*80)
            
            # Create a summary plot of accuracy across periods
            period_numbers = [r['period'] for r in results]
            accuracies = [r['accuracy_pct'] for r in results if r['accuracy_pct'] is not None]
            direction_accuracies = [r['direction_accuracy'] for r in results if r['direction_accuracy'] is not None]
            
            if accuracies:
                plt.figure(figsize=(10, 6))
                plt.bar(period_numbers, accuracies, alpha=0.7, label='Price Accuracy')
                
                if direction_accuracies and len(direction_accuracies) == len(period_numbers):
                    plt.plot(period_numbers, direction_accuracies, 'ro-', label='Direction Accuracy')
                
                plt.axhline(y=aggregate_metrics['avg_accuracy'], color='b', linestyle='--', label=f'Avg Accuracy: {aggregate_metrics["avg_accuracy"]:.2f}%')
                
                plt.title(f'Prediction Accuracy Across {len(results)} Backtest Periods ({prediction_days}-day horizon)')
                plt.xlabel('Backtest Period')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if save_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.output_dir, f'backtest_summary_{timestamp}.png')
                    plt.savefig(output_path)
                    logging.info(f"Saved backtest summary plot to {output_path}")
                
                figures.append(plt.gcf())
                plt.close()
            
            # Save results
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(self.output_dir, f'backtest_results_{timestamp}.csv')
                
                # Convert results to DataFrame
                results_df = pd.DataFrame(results)
                results_df.to_csv(results_path)
                logging.info(f"Saved backtest results to {results_path}")
                
                # Also save aggregate metrics
                aggregates_path = os.path.join(self.output_dir, f'backtest_aggregates_{timestamp}.csv')
                pd.DataFrame([aggregate_metrics]).to_csv(aggregates_path)
            
            return {
                'individual_results': results,
                'aggregate_metrics': aggregate_metrics,
                'figures': figures
            }
        
        return None