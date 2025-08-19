import torch
import numpy as np
import pandas as pd
import logging
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Configure tf-keras compatibility warning
import warnings
warnings.filterwarnings("ignore", message=".*module 'keras.api.*")

# Setup TF/transformers imports with fallback
HAS_TRANSFORMERS = False
try:
    # Try to import tf-keras first if it exists
    try:
        import tf_keras
    except ImportError:
        pass

    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        TrainingArguments, 
        Trainer, 
        pipeline,
        BertTokenizer,
        BertForSequenceClassification,
        RobertaTokenizer,
        RobertaForSequenceClassification,
        DistilBertTokenizer,
        DistilBertForSequenceClassification
    )
    from datasets import Dataset, DatasetDict
    HAS_TRANSFORMERS = True
except Exception as e:
    logging.warning(f"Transformers library not fully available: {e}")
    logging.warning("Running with limited sentiment analysis functionality")

# Import text processor with appropriate name
try:
    from src.preprocessing.text_processor import StockTextProcessor as TextProcessor
except ImportError:
    logging.warning("StockTextProcessor not found, importing as TextProcessor may fail")
    from src.preprocessing.text_processor import TextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSentimentAnalyzer:
    """
    Class for advanced sentiment analysis using transformer models
    (BERT, RoBERTa, DistilBERT, FinBERT)
    """
    
    def __init__(self, model_name=None, text_processor=None, device=None):
        """
        Initialize the advanced sentiment analyzer
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the transformer model to use. 
            Options: 'bert', 'roberta', 'distilbert', 'finbert'
            If None, uses 'distilbert' for better performance on lower resources
        text_processor : TextProcessor, optional
            Text processor for additional preprocessing (optional for transformers)
        device : str, optional
            Device to use for model inference ('cuda', 'cpu', or None for auto)
        """
        self.text_processor = text_processor
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Set model configuration
        self.supported_models = {
            'bert': {
                'name': 'bert-base-uncased',
                'tokenizer_class': BertTokenizer,
                'model_class': BertForSequenceClassification
            },
            'roberta': {
                'name': 'roberta-base',
                'tokenizer_class': RobertaTokenizer,
                'model_class': RobertaForSequenceClassification
            },
            'distilbert': {
                'name': 'distilbert-base-uncased',
                'tokenizer_class': DistilBertTokenizer,
                'model_class': DistilBertForSequenceClassification
            },
            'finbert': {
                'name': 'ProsusAI/finbert',  # Financial domain BERT
                'tokenizer_class': BertTokenizer,  # FinBERT uses BERT tokenizer
                'model_class': BertForSequenceClassification
            }
        }
        
        # Default to distilbert if no model specified (lighter and faster)
        if model_name is None:
            model_name = 'distilbert'
            
        if model_name not in self.supported_models:
            logger.warning(f"Model {model_name} not supported, falling back to distilbert")
            model_name = 'distilbert'
        
        self.model_name = model_name
        self.model_config = self.supported_models[model_name]
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.trained = False
        
        # Label mapping (for sentiment classes)
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def load_pretrained_model(self, num_labels=3):
        """
        Load a pre-trained transformer model
        
        Parameters:
        -----------
        num_labels : int
            Number of sentiment classes
        """
        try:
            # Load tokenizer
            logger.info(f"Loading {self.model_name} tokenizer: {self.model_config['name']}")
            self.tokenizer = self.model_config['tokenizer_class'].from_pretrained(
                self.model_config['name']
            )
            
            # Load model
            logger.info(f"Loading {self.model_name} model: {self.model_config['name']}")
            self.model = self.model_config['model_class'].from_pretrained(
                self.model_config['name'],
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            ).to(self.device)
            
            # Create sentiment pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_data(self, texts, labels=None):
        """
        Preprocess data for transformer models
        
        Parameters:
        -----------
        texts : list
            List of texts to preprocess
        labels : list, optional
            List of labels
            
        Returns:
        --------
        datasets.Dataset
            Preprocessed dataset
        """
        # Convert texts to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Convert labels to list if needed and provided
        if labels is not None and isinstance(labels, pd.Series):
            labels = labels.tolist()
            
        # Create dataset dictionary
        dataset_dict = {'text': texts}
        
        if labels is not None:
            # Convert string labels to ids
            label_ids = [self.label2id.get(label, 1) for label in labels]  # Default to neutral (1) if unknown
            dataset_dict['label'] = label_ids
            
        # Create dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        return dataset
    
    def tokenize_data(self, dataset):
        """
        Tokenize data for transformer models
        
        Parameters:
        -----------
        dataset : datasets.Dataset
            Dataset to tokenize
            
        Returns:
        --------
        datasets.Dataset
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length",
                truncation=True,
                max_length=128
            )
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_model(self, train_texts, train_labels, val_texts=None, val_labels=None, 
                    epochs=3, batch_size=16, learning_rate=5e-5, output_dir='models/sentiment'):
        """
        Fine-tune transformer model on custom data
        
        Parameters:
        -----------
        train_texts : list or pandas.Series
            Training texts
        train_labels : list or pandas.Series
            Training labels (as strings: 'negative', 'neutral', 'positive')
        val_texts : list or pandas.Series, optional
            Validation texts
        val_labels : list or pandas.Series, optional
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Training batch size
        learning_rate : float
            Learning rate for training
        output_dir : str
            Directory to save model checkpoints
            
        Returns:
        --------
        dict
            Training results
        """
        try:
            # If model not loaded yet, load it
            if self.model is None or self.tokenizer is None:
                self.load_pretrained_model()
            
            # Check if inputs are pandas Series and convert to list if needed
            if isinstance(train_texts, pd.Series):
                train_texts = train_texts.tolist()
            if isinstance(train_labels, pd.Series):
                train_labels = train_labels.tolist()
                
            # Create training dataset
            train_dataset = self.preprocess_data(train_texts, train_labels)
            
            # If validation data is provided, create validation dataset
            if val_texts is not None and val_labels is not None:
                val_dataset = self.preprocess_data(val_texts, val_labels)
                dataset = DatasetDict({
                    'train': train_dataset,
                    'validation': val_dataset
                })
            else:
                # If no validation data, split train data
                dataset = train_dataset.train_test_split(test_size=0.1)
            
            # Tokenize datasets
            tokenized_datasets = dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"], 
                    padding="max_length",
                    truncation=True,
                    max_length=128
                ),
                batched=True
            )
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                learning_rate=learning_rate,
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=10,
                eval_steps=100,
                save_steps=1000,
                evaluation_strategy="steps"
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
            )
            
            # Train model
            logger.info("Starting model training")
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model(os.path.join(output_dir, "final_model"))
            self.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
            
            # Update the loaded model and create a new pipeline
            self.model = trainer.model
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.trained = True
            logger.info("Model training completed")
            
            # Compute evaluation metrics
            eval_result = {}
            if "validation" in tokenized_datasets:
                logger.info("Evaluating model")
                eval_result = trainer.evaluate()
            
            return {
                "train_result": train_result,
                "eval_result": eval_result
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def load_finetuned_model(self, model_dir):
        """
        Load a fine-tuned model from disk
        
        Parameters:
        -----------
        model_dir : str
            Directory containing the fine-tuned model
        """
        try:
            logger.info(f"Loading fine-tuned model from {model_dir}")
            
            # Load tokenizer from saved model
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Load model from saved model
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
            
            # Create sentiment pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.trained = True
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise
    
    def analyze_sentiment(self, text, preprocess=True):
        """
        Analyze sentiment of a single text
        
        Parameters:
        -----------
        text : str
            Text to analyze
        preprocess : bool
            Whether to apply additional preprocessing using the text processor
            
        Returns:
        --------
        dict
            Sentiment analysis results
        """
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            logger.info("Model not loaded yet, loading default model")
            self.load_pretrained_model()
            
        # Preprocess text if required and text processor is available
        if preprocess and self.text_processor is not None:
            # Extract ticker symbols before preprocessing
            tickers = self.text_processor.extract_stock_tickers(text)
            processed_text = self.text_processor.preprocess_text(text)
        else:
            tickers = []
            processed_text = text
        
        try:
            # Perform sentiment analysis
            start_time = time.time()
            result = self.pipeline(processed_text)[0]
            inference_time = time.time() - start_time
            
            # Transform result to standard format
            sentiment_result = {
                'sentiment': result['label'].lower(),
                'confidence': result['score'],
                'inference_time': inference_time
            }
            
            # Add ticker information if available
            if tickers:
                sentiment_result['tickers'] = tickers
                
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def batch_analyze(self, texts, batch_size=32, preprocess=True, show_progress=True):
        """
        Analyze sentiment for a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        batch_size : int
            Size of batches for processing
        preprocess : bool
            Whether to apply additional preprocessing using the text processor
        show_progress : bool
            Whether to show a progress bar
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment analysis results
        """
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            logger.info("Model not loaded yet, loading default model")
            self.load_pretrained_model()
            
        # Convert to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Process in batches to avoid memory issues
        results = []
        
        # Setup progress bar if requested
        iterator = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Preprocess batch if required and text processor is available
            if preprocess and self.text_processor is not None:
                # Process texts and extract tickers
                processed_batch = []
                batch_tickers = []
                
                for text in batch_texts:
                    tickers = self.text_processor.extract_stock_tickers(text)
                    processed_text = self.text_processor.preprocess_text(text)
                    processed_batch.append(processed_text)
                    batch_tickers.append(tickers)
            else:
                processed_batch = batch_texts
                batch_tickers = [[] for _ in batch_texts]
            
            try:
                # Perform sentiment analysis on batch
                # Perform sentiment analysis on batch
                start_time = time.time()
                batch_results = self.pipeline(processed_batch)
                inference_time = time.time() - start_time
                
                # Transform results to standard format
                for j, result in enumerate(batch_results):
                    sentiment_result = {
                        'sentiment': result['label'].lower(),
                        'confidence': result['score'],
                        'inference_time': inference_time / len(batch_texts)  # Average time per text
                    }
                    
                    # Add ticker information if available
                    if batch_tickers[j]:
                        sentiment_result['tickers'] = batch_tickers[j]
                        
                    # Add original text reference
                    sentiment_result['original_text'] = batch_texts[j]
                    
                    results.append(sentiment_result)
                    
            except Exception as e:
                logger.error(f"Error analyzing batch {i//batch_size}: {e}")
                # Add neutral results for failed batch
                for text in batch_texts:
                    results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'error': str(e),
                        'original_text': text
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def analyze_dataframe(self, df, text_column, result_prefix='sentiment_', batch_size=32):
        """
        Analyze sentiment for texts in a DataFrame column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to analyze
        text_column : str
            Name of column containing texts
        result_prefix : str
            Prefix for new columns with analysis results
        batch_size : int
            Size of batches for processing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sentiment analysis columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Analyze sentiment for each text
        sentiments = self.batch_analyze(df[text_column], batch_size=batch_size)
        
        # Add sentiment columns to result DataFrame
        result_df[f'{result_prefix}category'] = sentiments['sentiment']
        result_df[f'{result_prefix}confidence'] = sentiments['confidence']
        
        # Add tickers if available
        if 'tickers' in sentiments.columns:
            result_df[f'{result_prefix}tickers'] = sentiments['tickers']
        
        return result_df
    
    def evaluate_model(self, test_texts, test_labels, batch_size=32):
        """
        Evaluate model performance on test data
        
        Parameters:
        -----------
        test_texts : list or pandas.Series
            Test texts
        test_labels : list or pandas.Series
            Test labels (as strings: 'negative', 'neutral', 'positive')
        batch_size : int
            Size of batches for processing
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        # Convert to list if needed
        if isinstance(test_texts, pd.Series):
            test_texts = test_texts.tolist()
        if isinstance(test_labels, pd.Series):
            test_labels = test_labels.tolist()
            
        # Convert string labels to ids
        true_label_ids = [self.label2id.get(label, 1) for label in test_labels]  # Default to neutral (1) if unknown
        
        # Get predictions
        predictions = self.batch_analyze(test_texts, batch_size=batch_size)
        pred_labels = predictions['sentiment'].tolist()
        
        # Convert predicted labels to ids
        pred_label_ids = [self.label2id.get(label, 1) for label in pred_labels]
        
        # Calculate metrics
        accuracy = accuracy_score(true_label_ids, pred_label_ids)
        f1 = f1_score(true_label_ids, pred_label_ids, average='weighted')
        report = classification_report(true_label_ids, pred_label_ids, 
                                     target_names=list(self.id2label.values()),
                                     output_dict=True)
        conf_matrix = confusion_matrix(true_label_ids, pred_label_ids)
        
        # Create evaluation dictionary
        eval_results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        logger.info(f"Evaluation completed with accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return eval_results
    
    def plot_confusion_matrix(self, confusion_matrix):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        confusion_matrix : numpy.ndarray
            Confusion matrix to plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.id2label.values()),
                   yticklabels=list(self.id2label.values()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('models', f'{self.model_name}_confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def plot_confidence_distribution(self, predictions):
        """
        Plot distribution of prediction confidences
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            DataFrame with prediction results from batch_analyze
        """
        plt.figure(figsize=(10, 6))
        
        # Group by sentiment
        for sentiment in self.id2label.values():
            sent_preds = predictions[predictions['sentiment'] == sentiment]
            if len(sent_preds) > 0:
                sns.kdeplot(sent_preds['confidence'], label=sentiment)
        
        plt.title('Confidence Distribution by Sentiment Class')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        save_path = os.path.join('models', f'{self.model_name}_confidence_distribution.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def save_analyzer(self, filepath):
        """
        Save analyzer configuration (without the model) to a file
        Note: The model itself is not saved here, use save_model for that
        
        Parameters:
        -----------
        filepath : str
            File path to save analyzer to
        """
        try:
            # Create a copy of self without model and tokenizer to avoid serialization issues
            config = {
                'model_name': self.model_name,
                'device': self.device,
                'id2label': self.id2label,
                'label2id': self.label2id,
                'trained': self.trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(config, f)
            logger.info(f"Analyzer configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving analyzer configuration: {e}")
    
    def save_model(self, directory):
        """
        Save the model and tokenizer to directory
        
        Parameters:
        -----------
        directory : str
            Directory to save model to
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(directory)
            self.tokenizer.save_pretrained(directory)
            
            # Save analyzer configuration
            config_path = os.path.join(directory, 'analyzer_config.pkl')
            self.save_analyzer(config_path)
            
            logger.info(f"Model and tokenizer saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    @classmethod
    def load_analyzer(cls, config_filepath, model_directory=None):
        """
        Load analyzer from configuration file and optionally load model
        
        Parameters:
        -----------
        config_filepath : str
            File path to load analyzer configuration from
        model_directory : str, optional
            Directory to load model from
            
        Returns:
        --------
        AdvancedSentimentAnalyzer
            Loaded analyzer
        """
        try:
            # Load configuration
            with open(config_filepath, 'rb') as f:
                config = pickle.load(f)
            
            # Create analyzer
            analyzer = cls(model_name=config['model_name'], device=config['device'])
            analyzer.id2label = config['id2label']
            analyzer.label2id = config['label2id']
            analyzer.trained = config['trained']
            
            # Load model if directory provided
            if model_directory:
                analyzer.load_finetuned_model(model_directory)
            
            logger.info(f"Analyzer loaded from {config_filepath}")
            return analyzer
            
        except Exception as e:
            logger.error(f"Error loading analyzer: {e}")
            return cls()  # Return a new instance if loading fails


class FinBERTSentiment:
    """
    Specialized class for financial sentiment analysis using FinBERT
    FinBERT is a BERT model fine-tuned on financial text
    """
    
    def __init__(self, text_processor=None, device=None):
        """
        Initialize FinBERT sentiment analyzer
        
        Parameters:
        -----------
        text_processor : TextProcessor, optional
            Text processor for additional preprocessing
        device : str, optional
            Device to use for model inference ('cuda', 'cpu', or None for auto)
        """
        self.text_processor = text_processor
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Set model configuration
        self.model_name = 'finbert'
        self.model_path = 'ProsusAI/finbert'
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # FinBERT specific label mapping (different from general model)
        self.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        self.label2id = {'positive': 0, 'negative': 1, 'neutral': 2}
    
    def load_model(self):
        """
        Load FinBERT model
        """
        try:
            # Load tokenizer
            logger.info("Loading FinBERT tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            
            # Load model
            logger.info("Loading FinBERT model")
            self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            
            # Create sentiment pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def analyze_sentiment(self, text, preprocess=True):
        """
        Analyze financial sentiment of a single text using FinBERT
        
        Parameters:
        -----------
        text : str
            Text to analyze
        preprocess : bool
            Whether to apply additional preprocessing using the text processor
            
        Returns:
        --------
        dict
            Financial sentiment analysis results
        """
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            logger.info("FinBERT model not loaded yet, loading now")
            self.load_model()
            
        # Preprocess text if required and text processor is available
        if preprocess and self.text_processor is not None:
            # Extract ticker symbols before preprocessing
            tickers = self.text_processor.extract_stock_tickers(text)
            processed_text = self.text_processor.preprocess_text(text)
        else:
            tickers = []
            processed_text = text
        
        try:
            # Perform sentiment analysis
            start_time = time.time()
            result = self.pipeline(processed_text)[0]
            inference_time = time.time() - start_time
            
            # Map label to standard format (FinBERT has its own labeling)
            label = result['label'].lower()
            
            # Transform result to standard format
            sentiment_result = {
                'sentiment': label,
                'confidence': result['score'],
                'inference_time': inference_time
            }
            
            # Add ticker information if available
            if tickers:
                sentiment_result['tickers'] = tickers
                
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment with FinBERT: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def batch_analyze(self, texts, batch_size=32, preprocess=True, show_progress=True):
        """
        Analyze financial sentiment for a batch of texts using FinBERT
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        batch_size : int
            Size of batches for processing
        preprocess : bool
            Whether to apply additional preprocessing using the text processor
        show_progress : bool
            Whether to show a progress bar
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with financial sentiment analysis results
        """
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            logger.info("FinBERT model not loaded yet, loading now")
            self.load_model()
            
        # Convert to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Process in batches to avoid memory issues
        results = []
        
        # Setup progress bar if requested
        iterator = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Preprocess batch if required and text processor is available
            if preprocess and self.text_processor is not None:
                # Process texts and extract tickers
                processed_batch = []
                batch_tickers = []
                
                for text in batch_texts:
                    tickers = self.text_processor.extract_stock_tickers(text)
                    processed_text = self.text_processor.preprocess_text(text)
                    processed_batch.append(processed_text)
                    batch_tickers.append(tickers)
            else:
                processed_batch = batch_texts
                batch_tickers = [[] for _ in batch_texts]
            
            try:
                # Perform sentiment analysis on batch
                start_time = time.time()
                batch_results = self.pipeline(processed_batch)
                inference_time = time.time() - start_time
                
                # Transform results to standard format
                for j, result in enumerate(batch_results):
                    # Map label to standard format
                    # Map label to standard format
                    label = result['label'].lower()
                    
                    sentiment_result = {
                        'sentiment': label,
                        'confidence': result['score'],
                        'inference_time': inference_time / len(batch_texts)  # Average time per text
                    }
                    
                    # Add ticker information if available
                    if batch_tickers[j]:
                        sentiment_result['tickers'] = batch_tickers[j]
                        
                    # Add original text reference
                    sentiment_result['original_text'] = batch_texts[j]
                    
                    results.append(sentiment_result)
                    
            except Exception as e:
                logger.error(f"Error analyzing batch {i//batch_size}: {e}")
                # Add neutral results for failed batch
                for text in batch_texts:
                    results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'error': str(e),
                        'original_text': text
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def analyze_dataframe(self, df, text_column, result_prefix='finbert_', batch_size=32):
        """
        Analyze financial sentiment for texts in a DataFrame column using FinBERT
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to analyze
        text_column : str
            Name of column containing texts
        result_prefix : str
            Prefix for new columns with analysis results
        batch_size : int
            Size of batches for processing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added financial sentiment analysis columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Analyze sentiment for each text
        sentiments = self.batch_analyze(df[text_column], batch_size=batch_size)
        
        # Add sentiment columns to result DataFrame
        result_df[f'{result_prefix}category'] = sentiments['sentiment']
        result_df[f'{result_prefix}confidence'] = sentiments['confidence']
        
        # Calculate financial sentiment score (-1 to 1 scale)
        # Where positive = 1, negative = -1, neutral = 0
        sentiment_score = sentiments['sentiment'].map({
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0
        }) * sentiments['confidence']
        
        result_df[f'{result_prefix}score'] = sentiment_score
        
        # Add tickers if available
        if 'tickers' in sentiments.columns:
            result_df[f'{result_prefix}tickers'] = sentiments['tickers']
        
        return result_df
    
    def analyze_ticker_sentiment(self, texts, ticker_symbol=None):
        """
        Analyze sentiment specifically for a ticker symbol
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        ticker_symbol : str, optional
            Ticker symbol to filter for. If None, analyzes all tickers found.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with financial sentiment analysis for the specified ticker
        """
        # Analyze all texts
        results = self.batch_analyze(texts)
        
        # Filter for texts containing the ticker if specified
        if ticker_symbol:
            filtered_results = []
            
            for _, row in results.iterrows():
                if 'tickers' in row and isinstance(row['tickers'], list):
                    if ticker_symbol.upper() in [t.upper() for t in row['tickers']]:
                        filtered_results.append(row)
                        
            results = pd.DataFrame(filtered_results) if filtered_results else pd.DataFrame()
                
        # Compute aggregate sentiment
        if not results.empty:
            pos_count = (results['sentiment'] == 'positive').sum()
            neg_count = (results['sentiment'] == 'negative').sum()
            neu_count = (results['sentiment'] == 'neutral').sum()
            
            total = len(results)
            pos_pct = pos_count / total * 100 if total > 0 else 0
            neg_pct = neg_count / total * 100 if total > 0 else 0
            neu_pct = neu_count / total * 100 if total > 0 else 0
            
            # Add summary to results
            results.attrs['summary'] = {
                'ticker': ticker_symbol,
                'total_mentions': total,
                'positive_count': pos_count,
                'negative_count': neg_count,
                'neutral_count': neu_count,
                'positive_percentage': pos_pct,
                'negative_percentage': neg_pct,
                'neutral_percentage': neu_pct,
                'sentiment_ratio': pos_count / max(neg_count, 1)  # Positive to negative ratio
            }
            
        return results
    
    def plot_ticker_sentiment(self, ticker_results):
        """
        Plot sentiment distribution for a ticker
        
        Parameters:
        -----------
        ticker_results : pandas.DataFrame
            Results from analyze_ticker_sentiment
        """
        if ticker_results.empty or 'summary' not in ticker_results.attrs:
            logger.warning("No ticker sentiment data available to plot")
            return
            
        summary = ticker_results.attrs['summary']
        ticker = summary['ticker'] if summary['ticker'] else "All Tickers"
        
        # Create sentiment distribution chart
        plt.figure(figsize=(10, 6))
        
        # Bar colors
        colors = ['#2ecc71', '#e74c3c', '#3498db']  # green, red, blue
        
        # Create bar chart
        sentiment_data = [
            summary['positive_percentage'], 
            summary['negative_percentage'], 
            summary['neutral_percentage']
        ]
        
        bars = plt.bar(['Positive', 'Negative', 'Neutral'], sentiment_data, color=colors)
        
        # Add counts as text on bars
        for i, bar in enumerate(bars):
            count = [
                summary['positive_count'], 
                summary['negative_count'], 
                summary['neutral_count']
            ][i]
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{count}', ha='center', va='bottom')
        
        plt.title(f'Sentiment Distribution for {ticker}')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, max(sentiment_data) * 1.2)  # Add some space for text
        
        # Add total mentions and sentiment ratio
        plt.figtext(0.15, 0.01, f"Total Mentions: {summary['total_mentions']}", 
                  fontsize=10, ha='left')
        plt.figtext(0.85, 0.01, f"Positive/Negative Ratio: {summary['sentiment_ratio']:.2f}", 
                  fontsize=10, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('analysis', f'sentiment_{ticker.lower().replace("$", "")}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def save_model(self, directory):
        """
        Save the FinBERT model to directory
        
        Parameters:
        -----------
        directory : str
            Directory to save model to
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(directory)
            self.tokenizer.save_pretrained(directory)
            
            logger.info(f"FinBERT model saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving FinBERT model: {e}")


# If script is run directly, demonstrate basic usage
if __name__ == "__main__":
    # Example texts
    example_texts = [
        "I am very bullish on $AAPL, their new products look amazing.",
        "Extremely bearish on $TSLA, the company is facing significant challenges.",
        "Not sure about $AMZN, waiting to see their earnings report."
    ]
    
    # Initialize FinBERT analyzer
    finbert = FinBERTSentiment()
    
    # Load model (if not in interactive mode, comment this out and it will load automatically)
    finbert.load_model()
    
    # Analyze texts
    print("FinBERT Analysis:")
    for text in example_texts:
        result = finbert.analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.4f}")
        if 'tickers' in result:
            print(f"Tickers: {result['tickers']}")
        print()
    
    # Analyze batch
    batch_results = finbert.batch_analyze(example_texts)
    
    # Print summary
    print("\nBatch Analysis Summary:")
    sentiment_counts = batch_results['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"{sentiment}: {count} ({count/len(batch_results)*100:.1f}%)")