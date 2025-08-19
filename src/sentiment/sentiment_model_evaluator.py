"""
Sentiment Model Evaluator
Provides functionality to compare and evaluate different sentiment analysis approaches
using metrics like accuracy, precision, recall, F1 score, and custom financial metrics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, mean_absolute_error)
from sklearn.model_selection import train_test_split, cross_val_score
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple, Union, Any, Optional
import json

# Local imports
from src.sentiment.basic_sentiment import BasicSentimentAnalyzer
from src.sentiment.advanced_sentiment import AdvancedSentimentAnalyzer, FinBERTSentiment
from src.sentiment.emotion_sentiment import EmotionalSentimentAnalyzer
from src.sentiment.advanced_emotion_sentiment import EnhancedEmotionalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class SentimentModelEvaluator:
    """
    Class for evaluating and comparing different sentiment analysis models
    """
    def __init__(self, output_dir='results/sentiment_evaluation'):
        """
        Initialize sentiment model evaluator
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize model instances
        self.models = {}
    
    def add_model(self, model_name: str, model_instance: Any):
        """
        Add a model to be evaluated
        
        Parameters:
        -----------
        model_name : str
            Name to identify the model
        model_instance : object
            Sentiment analyzer instance
        """
        self.models[model_name] = model_instance
        logger.info(f"Model '{model_name}' added to evaluator")
    
    def load_standard_models(self):
        """
        Load standard sentiment models for comparison
        """
        try:
            # Basic sentiment analyzer
            basic_model = BasicSentimentAnalyzer()
            self.add_model("Basic Sentiment", basic_model)
            
            # FinBERT sentiment analyzer
            finbert_model = FinBERTSentiment()
            finbert_model.load_model()
            self.add_model("FinBERT", finbert_model)
            
            # Emotional sentiment analyzer
            emotional_model = EmotionalSentimentAnalyzer()
            try:
                emotional_model.load_emotion_model()
            except Exception as e:
                logger.warning(f"Could not load emotion model: {e}")
            self.add_model("Emotional Sentiment", emotional_model)
            
            # Enhanced emotional analyzer
            enhanced_model = EnhancedEmotionalAnalyzer()
            try:
                enhanced_model.load_emotion_model()
                enhanced_model.load_fine_emotion_model()
            except Exception as e:
                logger.warning(f"Could not load enhanced emotion models: {e}")
            self.add_model("Enhanced Emotional", enhanced_model)
            
            # Advanced BERT-based analyzer
            advanced_model = AdvancedSentimentAnalyzer()
            advanced_model.load_pretrained_model()
            self.add_model("Advanced BERT", advanced_model)
            
            logger.info("Standard models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading standard models: {e}")
    
    def evaluate_text_classification(self, texts: List[str], labels: List[str]) -> pd.DataFrame:
        """
        Evaluate all models on text classification task
        
        Parameters:
        -----------
        texts : list of str
            List of texts to analyze
        labels : list of str
            Ground truth sentiment labels (positive, negative, neutral)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with evaluation metrics for each model
        """
        if not self.models:
            logger.error("No models loaded for evaluation")
            return pd.DataFrame()
        
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Get predictions based on model type
                predictions = self._get_model_predictions(model, texts)
                
                # Calculate metrics
                metrics = self._calculate_classification_metrics(labels, predictions, model_name)
                results.append(metrics)
                
                logger.info(f"Model {model_name} evaluation completed")
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
        
        # Create metrics DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by F1 score
        if not results_df.empty and 'f1' in results_df.columns:
            results_df = results_df.sort_values('f1', ascending=False)
        
        # Save results
        self._save_classification_results(results_df)
        
        return results_df
    
    def evaluate_sentiment_scores(self, texts: List[str], reference_scores: List[float]) -> pd.DataFrame:
        """
        Evaluate all models on continuous sentiment scoring task
        
        Parameters:
        -----------
        texts : list of str
            List of texts to analyze
        reference_scores : list of float
            Ground truth sentiment scores (-1 to +1 scale)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with evaluation metrics for each model
        """
        if not self.models:
            logger.error("No models loaded for evaluation")
            return pd.DataFrame()
        
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Get sentiment scores based on model type
                scores = self._get_model_scores(model, texts)
                
                # Calculate regression metrics
                metrics = self._calculate_regression_metrics(reference_scores, scores, model_name)
                results.append(metrics)
                
                logger.info(f"Model {model_name} evaluation completed")
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
        
        # Create metrics DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by MAE (lower is better)
        if not results_df.empty and 'mae' in results_df.columns:
            results_df = results_df.sort_values('mae')
        
        # Save results
        self._save_regression_results(results_df)
        
        return results_df
    
    def evaluate_market_direction_accuracy(self, texts: List[str], 
                                         actual_moves: List[float],
                                         prediction_window: int = 1) -> pd.DataFrame:
        """
        Evaluate models on predicting market direction (up/down)
        
        Parameters:
        -----------
        texts : list of str
            List of financial texts to analyze
        actual_moves : list of float
            Subsequent market price changes
        prediction_window : int
            Number of periods to look forward for price movement
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with market prediction metrics for each model
        """
        if not self.models:
            logger.error("No models loaded for evaluation")
            return pd.DataFrame()
        
        # Convert price moves to binary direction (1 for up, 0 for down)
        actual_directions = [1 if move > 0 else 0 for move in actual_moves]
        
        results = []
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating market direction accuracy for model: {model_name}")
            
            try:
                # Get sentiment scores
                scores = self._get_model_scores(model, texts)
                
                # Convert sentiment scores to predicted directions
                # Positive sentiment -> market up, Negative sentiment -> market down
                predicted_directions = [1 if score > 0 else 0 for score in scores]
                
                # Calculate direction accuracy metrics
                accuracy = accuracy_score(actual_directions, predicted_directions)
                precision = precision_score(actual_directions, predicted_directions, zero_division=0)
                recall = recall_score(actual_directions, predicted_directions, zero_division=0)
                f1 = f1_score(actual_directions, predicted_directions, zero_division=0)
                
                # Calculate correlation between sentiment and price moves
                correlation = np.corrcoef(scores, actual_moves)[0, 1]
                
                metrics = {
                    'model': model_name,
                    'direction_accuracy': accuracy,
                    'direction_precision': precision,
                    'direction_recall': recall,
                    'direction_f1': f1,
                    'sentiment_price_correlation': correlation,
                    'prediction_window': prediction_window
                }
                
                results.append(metrics)
                logger.info(f"Market direction evaluation for {model_name} completed")
            except Exception as e:
                logger.error(f"Error evaluating market direction for model {model_name}: {e}")
        
        # Create metrics DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by direction accuracy
        if not results_df.empty and 'direction_accuracy' in results_df.columns:
            results_df = results_df.sort_values('direction_accuracy', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"market_direction_evaluation_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        
        # Plot results
        self._plot_market_direction_results(results_df, timestamp)
        
        return results_df
    
    def compare_emotion_detection(self, texts: List[str], reference_emotions: Optional[List[Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Compare emotion detection capabilities of different models
        
        Parameters:
        -----------
        texts : list of str
            List of texts to analyze for emotions
        reference_emotions : list of dict, optional
            Ground truth emotion labels/scores if available
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with emotion detection metrics for each model
        """
        emotion_capable_models = {name: model for name, model in self.models.items() 
                                if isinstance(model, (EmotionalSentimentAnalyzer, EnhancedEmotionalAnalyzer))}
        
        if not emotion_capable_models:
            logger.error("No emotion-capable models loaded for evaluation")
            return pd.DataFrame()
        
        results = []
        emotion_data = {}
        
        for model_name, model in emotion_capable_models.items():
            logger.info(f"Evaluating emotion detection for model: {model_name}")
            
            try:
                # Get emotion results
                emotions_list = []
                
                for text in texts:
                    if isinstance(model, EnhancedEmotionalAnalyzer):
                        result = model.analyze_sentiment_advanced(text, include_fine_emotions=True)
                        if 'fine_emotions' in result:
                            emotions_list.append(result['fine_emotions'])
                        elif 'emotions' in result:
                            emotions_list.append(result['emotions'])
                    else:
                        result = model.analyze_sentiment(text)
                        if 'emotions' in result:
                            emotions_list.append(result['emotions'])
                
                # Calculate emotion coverage and diversity
                emotion_types = set()
                for emo_dict in emotions_list:
                    if emo_dict:
                        emotion_types.update(emo_dict.keys())
                
                # Store for plotting
                emotion_data[model_name] = {
                    'all_emotions': emotions_list,
                    'emotion_types': list(emotion_types)
                }
                
                # Prepare metrics
                metrics = {
                    'model': model_name,
                    'emotion_types_count': len(emotion_types),
                    'emotion_types': ', '.join(sorted(emotion_types)),
                    'emotions_detected_ratio': sum(1 for e in emotions_list if e) / len(texts),
                }
                
                # Add comparison with reference if available
                if reference_emotions:
                    # Add reference comparison metrics here
                    pass
                
                results.append(metrics)
                logger.info(f"Emotion detection evaluation for {model_name} completed")
            except Exception as e:
                logger.error(f"Error evaluating emotion detection for model {model_name}: {e}")
        
        # Create metrics DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by emotion types count
        if not results_df.empty and 'emotion_types_count' in results_df.columns:
            results_df = results_df.sort_values('emotion_types_count', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"emotion_detection_evaluation_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        
        # Plot emotion distribution by model
        self._plot_emotion_comparison(emotion_data, timestamp)
        
        return results_df
    
    def _get_model_predictions(self, model, texts):
        """Get sentiment predictions based on model type"""
        predictions = []
        
        for text in texts:
            try:
                if isinstance(model, BasicSentimentAnalyzer):
                    result = model.analyze_sentiment(text)
                    predictions.append(result.get('sentiment', 'neutral'))
                
                elif isinstance(model, FinBERTSentiment):
                    result = model.analyze_sentiment(text)
                    predictions.append(result.get('sentiment', 'neutral'))
                
                elif isinstance(model, AdvancedSentimentAnalyzer):
                    result = model.analyze_sentiment(text)
                    predictions.append(result.get('sentiment', 'neutral'))
                
                elif isinstance(model, EmotionalSentimentAnalyzer):
                    result = model.analyze_sentiment(text)
                    predictions.append(result.get('sentiment_label', 'neutral'))
                
                elif isinstance(model, EnhancedEmotionalAnalyzer):
                    result = model.analyze_sentiment_advanced(text)
                    predictions.append(result.get('sentiment_label', 'neutral'))
                
                else:
                    # Generic handling
                    if hasattr(model, 'analyze_sentiment'):
                        result = model.analyze_sentiment(text)
                        if isinstance(result, dict):
                            # Try to find sentiment in result keys
                            for key in ['sentiment', 'sentiment_label', 'label']:
                                if key in result:
                                    predictions.append(result[key])
                                    break
                            else:
                                predictions.append('neutral')
                        else:
                            predictions.append('neutral')
                    else:
                        logger.warning(f"Unsupported model type: {type(model)}")
                        predictions.append('neutral')
            
            except Exception as e:
                logger.error(f"Error getting prediction for text: {e}")
                predictions.append('neutral')
        
        return predictions
    
    def _get_model_scores(self, model, texts):
        """Get sentiment scores based on model type"""
        scores = []
        
        for text in texts:
            try:
                if isinstance(model, BasicSentimentAnalyzer):
                    result = model.analyze_sentiment(text)
                    scores.append(result.get('compound_score', 0.0))
                
                elif isinstance(model, FinBERTSentiment):
                    result = model.analyze_sentiment(text)
                    scores.append(result.get('score', 0.0))
                
                elif isinstance(model, AdvancedSentimentAnalyzer):
                    result = model.analyze_sentiment(text)
                    # Convert categorical to numerical if needed
                    if 'score' in result:
                        scores.append(result['score'])
                    else:
                        sentiment = result.get('sentiment', 'neutral')
                        score = 0.0
                        if sentiment == 'positive':
                            score = 0.5
                        elif sentiment == 'negative':
                            score = -0.5
                        scores.append(score)
                
                elif isinstance(model, EmotionalSentimentAnalyzer):
                    result = model.analyze_sentiment(text)
                    scores.append(result.get('sentiment_score', 0.0))
                
                elif isinstance(model, EnhancedEmotionalAnalyzer):
                    result = model.analyze_sentiment_advanced(text)
                    scores.append(result.get('sentiment_score', 0.0))
                
                else:
                    # Generic handling
                    if hasattr(model, 'analyze_sentiment'):
                        result = model.analyze_sentiment(text)
                        if isinstance(result, dict):
                            # Try to find score in result keys
                            for key in ['sentiment_score', 'score', 'compound', 'compound_score']:
                                if key in result:
                                    scores.append(result[key])
                                    break
                            else:
                                scores.append(0.0)
                        else:
                            scores.append(0.0)
                    else:
                        logger.warning(f"Unsupported model type: {type(model)}")
                        scores.append(0.0)
            
            except Exception as e:
                logger.error(f"Error getting score for text: {e}")
                scores.append(0.0)
        
        return scores
    
    def _calculate_classification_metrics(self, true_labels, predicted_labels, model_name):
        """Calculate classification metrics"""
        # Map strings to standard categories if needed
        standardized_true = []
        standardized_pred = []
        
        # Define standardization mapping
        label_mapping = {
            'positive': 'positive',
            'very_positive': 'positive',
            'negative': 'negative',
            'very_negative': 'negative',
            'neutral': 'neutral',
            'bullish': 'positive',
            'bearish': 'negative'
        }
        
        for t, p in zip(true_labels, predicted_labels):
            standardized_true.append(label_mapping.get(t.lower(), t.lower()))
            standardized_pred.append(label_mapping.get(p.lower(), p.lower()))
        
        # Calculate metrics
        accuracy = accuracy_score(standardized_true, standardized_pred)
        
        # Handle potential errors with precision, recall, f1 if there are missing classes
        try:
            precision = precision_score(standardized_true, standardized_pred, average='weighted', zero_division=0)
            recall = recall_score(standardized_true, standardized_pred, average='weighted', zero_division=0)
            f1 = f1_score(standardized_true, standardized_pred, average='weighted', zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating precision/recall metrics: {e}")
            precision, recall, f1 = 0, 0, 0
        
        # Get classification report as dict
        try:
            report = classification_report(standardized_true, standardized_pred, output_dict=True)
        except Exception:
            report = {}
        
        # Extract per-class metrics
        class_precision = {}
        class_recall = {}
        
        for cls in set(standardized_true):
            if cls in report:
                class_precision[f'precision_{cls}'] = report[cls]['precision']
                class_recall[f'recall_{cls}'] = report[cls]['recall']
        
        # Combine all metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **class_precision,
            **class_recall
        }
        
        return metrics
    
    def _calculate_regression_metrics(self, true_scores, predicted_scores, model_name):
        """Calculate regression metrics for sentiment scoring"""
        # Convert to numpy arrays
        true_array = np.array(true_scores)
        pred_array = np.array(predicted_scores)
        
        # Calculate metrics
        mae = mean_absolute_error(true_array, pred_array)
        
        # Calculate correlation
        correlation = np.corrcoef(true_array, pred_array)[0, 1]
        
        # Calculate directional accuracy (if scores have the same sign)
        directional_matches = np.sum((true_array > 0) == (pred_array > 0))
        directional_accuracy = directional_matches / len(true_array)
        
        # Combine all metrics
        metrics = {
            'model': model_name,
            'mae': mae,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy
        }
        
        return metrics
    
    def _save_classification_results(self, results_df):
        """Save classification evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        filepath = os.path.join(self.output_dir, f"sentiment_classification_evaluation_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        logger.info(f"Classification results saved to {filepath}")
        
        # Plot results
        self._plot_classification_results(results_df, timestamp)
    
    def _save_regression_results(self, results_df):
        """Save regression evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        filepath = os.path.join(self.output_dir, f"sentiment_scoring_evaluation_{timestamp}.csv")
        results_df.to_csv(filepath, index=False)
        logger.info(f"Regression results saved to {filepath}")
        
        # Plot results
        self._plot_regression_results(results_df, timestamp)
    
    def _plot_classification_results(self, results_df, timestamp):
        """Plot classification evaluation results"""
        plt.figure(figsize=(14, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(results_df))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                offset = width * (i - 1.5)
                ax.bar(x + offset, results_df[metric], width, label=metric.capitalize())
        
        ax.set_title('Sentiment Classification Models Comparison', fontsize=16)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, f"sentiment_classification_comparison_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Classification comparison plot saved to {filepath}")
    
    def _plot_regression_results(self, results_df, timestamp):
        """Plot regression evaluation results"""
        plt.figure(figsize=(14, 8))
        
        metrics = ['mae', 'correlation', 'directional_accuracy']
        metrics = [m for m in metrics if m in results_df.columns]
        
        # Create figure with two subplots (one for MAE, one for correlation)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot MAE (lower is better)
        if 'mae' in metrics:
            ax1.bar(results_df['model'], results_df['mae'], color='skyblue')
            ax1.set_title('Mean Absolute Error by Model', fontsize=14)
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('MAE (lower is better)', fontsize=12)
            ax1.set_xticklabels(results_df['model'], rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot correlation (higher is better)
        if 'correlation' in metrics:
            correlation_data = results_df[['model', 'correlation']]
            correlation_data = correlation_data.sort_values('correlation', ascending=False)
            
            bars = ax2.bar(correlation_data['model'], correlation_data['correlation'], color='lightgreen')
            ax2.set_title('Correlation with Reference Scores', fontsize=14)
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Correlation (higher is better)', fontsize=12)
            ax2.set_xticklabels(correlation_data['model'], rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, f"sentiment_scoring_comparison_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Regression comparison plot saved to {filepath}")
    
    def _plot_market_direction_results(self, results_df, timestamp):
        """Plot market direction prediction results"""
        plt.figure(figsize=(14, 8))
        
        metrics = ['direction_accuracy', 'direction_precision', 'direction_recall', 'direction_f1']
        metrics = [m for m in metrics if m in results_df.columns]
        
        x = np.arange(len(results_df))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, metric in enumerate(metrics):
            offset = width * (i - len(metrics)/2 + 0.5)
            ax.bar(x + offset, results_df[metric], width, label=metric.replace('direction_', '').capitalize())
        
        ax.set_title(f'Market Direction Prediction (Window: {results_df["prediction_window"].iloc[0]})', fontsize=16)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, f"market_direction_comparison_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Market direction comparison plot saved to {filepath}")
        
        # Also plot correlation as a separate chart if available
        if 'sentiment_price_correlation' in results_df.columns:
            plt.figure(figsize=(12, 6))
            
            correlation_data = results_df[['model', 'sentiment_price_correlation']]
            correlation_data = correlation_data.sort_values('sentiment_price_correlation', ascending=False)
            
            plt.bar(correlation_data['model'], correlation_data['sentiment_price_correlation'], color='lightblue')
            plt.title('Sentiment-Price Movement Correlation', fontsize=14)
            plt.xlabel('Model', fontsize=12)
            plt.ylabel('Correlation Coefficient', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            corr_filepath = os.path.join(self.output_dir, f"sentiment_price_correlation_{timestamp}.png")
            plt.savefig(corr_filepath, dpi=300, bbox_inches='tight')
    
    def _plot_emotion_comparison(self, emotion_data, timestamp):
        """Plot emotion detection comparison"""
        # Get common emotions across models
        all_emotions = set()
        for model_data in emotion_data.values():
            all_emotions.update(model_data['emotion_types'])
        
        common_emotions = sorted(list(all_emotions))
        
        # Create emotion frequency data
        emotion_freq_data = {}
        
        for model_name, model_data in emotion_data.items():
            # Initialize counters for each emotion
            emotion_counts = {emotion: 0 for emotion in common_emotions}
            
            # Count frequencies
            for emotion_dict in model_data['all_emotions']:
                if not emotion_dict:
                    continue
                    
                for emotion, score in emotion_dict.items():
                    if emotion in emotion_counts and score > 0:
                        emotion_counts[emotion] += 1
            
            # Convert to percentage
            total_texts = len(model_data['all_emotions'])
            emotion_freq = {e: (count / total_texts) * 100 for e, count in emotion_counts.items()}
            emotion_freq_data[model_name] = emotion_freq
        
        # Plot emotion comparison heatmap
        plt.figure(figsize=(16, 10))
        
        # Convert to DataFrame for easier plotting
        emotion_df = pd.DataFrame(emotion_freq_data).T
        
        # Sort columns by frequency
        column_sums = emotion_df.sum()
        emotion_df = emotion_df[column_sums.sort_values(ascending=False).index]
        
        # Plot heatmap
        ax = sns.heatmap(emotion_df, annot=True, fmt='.1f', cmap='viridis', 
                        linewidths=0.5, cbar_kws={'label': 'Frequency (%)'})
        
        plt.title('Emotion Detection Comparison Across Models', fontsize=16)
        plt.ylabel('Model', fontsize=12)
        plt.xlabel('Emotion Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        filepath = os.path.join(self.output_dir, f"emotion_detection_comparison_{timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Emotion detection comparison plot saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Sample financial texts
    sample_texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "We are downgrading this stock due to concerning cash flow issues.",
        "The market outlook is uncertain with mixed economic signals.",
        "I'm very bullish on this tech sector despite recent volatility.",
        "The CFO's resignation raises red flags about potential accounting problems."
    ]
    
    # Sample labels (positive, negative, neutral)
    sample_labels = ['positive', 'negative', 'neutral', 'positive', 'negative']
    
    # Sample sentiment scores (-1 to +1)
    sample_scores = [0.8, -0.6, 0.1, 0.9, -0.7]
    
    # Initialize evaluator
    evaluator = SentimentModelEvaluator()
    
    # Load standard models
    evaluator.load_standard_models()
    
    # Evaluate text classification
    classification_results = evaluator.evaluate_text_classification(sample_texts, sample_labels)
    print("\nSentiment Classification Results:")
    print(classification_results)
    
    # Evaluate sentiment scoring
    scoring_results = evaluator.evaluate_sentiment_scores(sample_texts, sample_scores)
    print("\nSentiment Scoring Results:")
    print(scoring_results)
    
    # Compare emotion detection
    emotion_results = evaluator.compare_emotion_detection(sample_texts)
    print("\nEmotion Detection Results:")
    print(emotion_results)