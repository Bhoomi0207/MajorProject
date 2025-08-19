import numpy as np
import pandas as pd
import logging
from textblob import TextBlob
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
import joblib
from src.preprocessing.text_processor import StockTextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicSentimentAnalyzer:
    """
    Class for basic sentiment analysis using rule-based and simple ML approaches
    """
    
    def __init__(self, text_processor=None):
        """
        Initialize analyzer with text processor
        
        Parameters:
        -----------
        text_processor : StockTextProcessor, optional
            Text processor to use for preprocessing. Creates a new one if None
        """
        self.text_processor = text_processor if text_processor is not None else StockTextProcessor()
        self.models = {}
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary with sentiment analysis results
        """
        # Preprocess text if it doesn't appear to be preprocessed
        if '[TICKERS:' not in text:
            text = self.text_processor.clean_text(text)
            
        # Analyze using TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment category based on polarity
        if polarity >= 0.1:
            sentiment = 'positive'
        elif polarity <= -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def batch_analyze_textblob(self, texts):
        """
        Analyze sentiment for a batch of texts using TextBlob
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment_textblob(text)
            results.append(result)
        return pd.DataFrame(results)
    
    def analyze_dataframe_textblob(self, df, text_column, result_prefix='sentiment_'):
        """
        Analyze sentiment for texts in a DataFrame column using TextBlob
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to analyze
        text_column : str
            Name of column containing texts
        result_prefix : str
            Prefix for new columns with analysis results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sentiment analysis columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Analyze sentiment for each text
        sentiments = self.batch_analyze_textblob(df[text_column])
        
        # Add sentiment columns to result DataFrame
        result_df[f'{result_prefix}category'] = sentiments['sentiment']
        result_df[f'{result_prefix}polarity'] = sentiments['polarity']
        result_df[f'{result_prefix}subjectivity'] = sentiments['subjectivity']
        
        return result_df
    
    def train_sentiment_model(self, texts, labels, model_type='logistic_regression', vectorizer_type='tfidf'):
        """
        Train a sentiment classification model
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts for training
        labels : list or pandas.Series
            Sentiment labels (positive, negative, neutral)
        model_type : str
            Type of model to train ('logistic_regression', 'random_forest', 'naive_bayes', 'svm')
        vectorizer_type : str
            Type of vectorizer to use ('count', 'tfidf')
            
        Returns:
        --------
        dict
            Dictionary with trained model, vectorizer, and evaluation metrics
        """
        # Preprocess texts
        processed_texts = [self.text_processor.clean_text(text) for text in texts]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Create vectorizer
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(max_features=5000)
        else:  # tfidf
            vectorizer = TfidfVectorizer(max_features=5000)
        
        # Create model
        if model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'naive_bayes':
            model = MultinomialNB()
        elif model_type == 'svm':
            model = LinearSVC(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('model', model)
        ])
        
        # Train model
        logger.info(f"Training {model_type} model with {vectorizer_type} vectorizer")
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Create and store model information
        model_info = {
            'pipeline': pipeline,
            'vectorizer_type': vectorizer_type,
            'model_type': model_type,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'classes': pipeline.classes_ if hasattr(pipeline, 'classes_') else None
        }
        
        # Store model in instance
        self.models[f"{model_type}_{vectorizer_type}"] = model_info
        
        return model_info
    
    def predict_sentiment(self, text, model_key=None):
        """
        Predict sentiment for a text using a trained model
        
        Parameters:
        -----------
        text : str
            Text to analyze
        model_key : str, optional
            Key of model to use. If None, uses first available model or TextBlob
            
        Returns:
        --------
        dict
            Dictionary with predicted sentiment and confidence
        """
        # If no model key provided, use first available model
        if model_key is None and self.models:
            model_key = list(self.models.keys())[0]
        
        # If no models available, use TextBlob
        if model_key is None or model_key not in self.models:
            logger.info("No trained model available, using TextBlob")
            return self.analyze_sentiment_textblob(text)
        
        # Preprocess text
        processed_text = self.text_processor.clean_text(text)
        
        # Get model info
        model_info = self.models[model_key]
        pipeline = model_info['pipeline']
        
        # Get prediction
        try:
            # Predict sentiment category
            sentiment = pipeline.predict([processed_text])[0]
            
            # Get prediction probability if available
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba([processed_text])[0]
                confidence = np.max(probabilities)
            else:
                confidence = None
                
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'model_used': model_key
            }
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return self.analyze_sentiment_textblob(text)
    
    def batch_predict_sentiment(self, texts, model_key=None):
        """
        Predict sentiments for a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        model_key : str, optional
            Key of model to use. If None, uses first available model or TextBlob
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment predictions
        """
        results = []
        for text in texts:
            result = self.predict_sentiment(text, model_key)
            results.append(result)
        return pd.DataFrame(results)
    
    def analyze_dataframe(self, df, text_column, model_key=None, result_prefix='sentiment_'):
        """
        Analyze sentiment for texts in a DataFrame column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to analyze
        text_column : str
            Name of column containing texts
        model_key : str, optional
            Key of model to use. If None, uses first available model or TextBlob
        result_prefix : str
            Prefix for new columns with analysis results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sentiment analysis columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Analyze sentiment for each text
        sentiments = self.batch_predict_sentiment(df[text_column], model_key)
        
        # Add sentiment columns to result DataFrame
        result_df[f'{result_prefix}category'] = sentiments['sentiment']
        if 'confidence' in sentiments.columns:
            result_df[f'{result_prefix}confidence'] = sentiments['confidence']
        if 'model_used' in sentiments.columns:
            result_df[f'{result_prefix}model_used'] = sentiments['model_used']
        
        return result_df
    
    def evaluate_all_models(self, texts, labels):
        """
        Train and evaluate multiple models for comparison
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts for training and evaluation
        labels : list or pandas.Series
            Sentiment labels
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with evaluation results for each model
        """
        model_types = ['logistic_regression', 'random_forest', 'naive_bayes', 'svm']
        vectorizer_types = ['count', 'tfidf']
        
        results = []
        
        for model_type in model_types:
            for vectorizer_type in vectorizer_types:
                try:
                    model_info = self.train_sentiment_model(
                        texts, labels, model_type, vectorizer_type
                    )
                    
                    results.append({
                        'model_type': model_type,
                        'vectorizer_type': vectorizer_type,
                        'accuracy': model_info['accuracy']
                    })
                    
                except Exception as e:
                    logger.error(f"Error training {model_type} with {vectorizer_type}: {e}")
        
        return pd.DataFrame(results)
    
    def plot_model_comparison(self, evaluation_results):
        """
        Plot model comparison results
        
        Parameters:
        -----------
        evaluation_results : pandas.DataFrame
            DataFrame with evaluation results from evaluate_all_models
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y='accuracy', hue='vectorizer_type', data=evaluation_results)
        plt.title('Sentiment Model Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('models', 'model_comparison.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def plot_confusion_matrix(self, model_key):
        """
        Plot confusion matrix for a trained model
        
        Parameters:
        -----------
        model_key : str
            Key of model to plot
        """
        if model_key not in self.models:
            logger.error(f"Model key {model_key} not found")
            return
        
        model_info = self.models[model_key]
        conf_matrix = model_info['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=model_info['classes'],
                   yticklabels=model_info['classes'])
        plt.title(f'Confusion Matrix for {model_key}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('models', f'{model_key}_confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def save_model(self, model_key, filepath=None):
        """
        Save a trained model to file
        
        Parameters:
        -----------
        model_key : str
            Key of model to save
        filepath : str, optional
            Filepath to save model to. If None, generates a default path
        """
        if model_key not in self.models:
            logger.error(f"Model key {model_key} not found")
            return
        
        if filepath is None:
            os.makedirs('models', exist_ok=True)
            filepath = os.path.join('models', f'{model_key}_sentiment_model.pkl')
        
        try:
            joblib.dump(self.models[model_key], filepath)
            logger.info(f"Model {model_key} saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath, model_key=None):
        """
        Load a trained model from file
        
        Parameters:
        -----------
        filepath : str
            Filepath to load model from
        model_key : str, optional
            Key to store model under. If None, extracts from filename
        
        Returns:
        --------
        dict
            Loaded model info
        """
        try:
            model_info = joblib.load(filepath)
            
            # If no model key provided, extract from filename
            if model_key is None:
                basename = os.path.basename(filepath)
                model_key = basename.split('_sentiment_model.pkl')[0]
            
            self.models[model_key] = model_info
            logger.info(f"Model loaded from {filepath} and stored as {model_key}")
            
            return model_info
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def save_analyzer(self, filepath):
        """
        Save this analyzer to a file for later use
        
        Parameters:
        -----------
        filepath : str
            File path to save analyzer to
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Sentiment analyzer saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving sentiment analyzer: {e}")
    
    @classmethod
    def load_analyzer(cls, filepath):
        """
        Load analyzer from file
        
        Parameters:
        -----------
        filepath : str
            File path to load analyzer from
            
        Returns:
        --------
        BasicSentimentAnalyzer
            Loaded sentiment analyzer
        """
        try:
            with open(filepath, 'rb') as f:
                analyzer = pickle.load(f)
            logger.info(f"Sentiment analyzer loaded from {filepath}")
            return analyzer
        except Exception as e:
            logger.error(f"Error loading sentiment analyzer: {e}")
            return cls()  # Return a new instance if loading fails


class StockMarketLexicon:
    """
    Class for managing a financial/stock market-specific lexicon for sentiment analysis
    """
    
    def __init__(self):
        """
        Initialize with built-in financial lexicon
        """
        # Positive and negative financial terms
        self.positive_terms = set([
            'bullish', 'buy', 'calls', 'upside', 'long', 'growth', 'rally', 
            'outperform', 'upgrade', 'strong', 'green', 'higher', 'gain', 'gains',
            'profit', 'profitable', 'positive', 'beat', 'exceeds', 'exceed', 
            'strong', 'strength', 'stronger', 'support', 'supported', 'supporting',
            'breakout', 'ramp', 'recover', 'recovery', 'surge', 'surging', 
            'uptrend', 'promising', 'opportunity', 'opportunities'
        ])
        
        self.negative_terms = set([
            'bearish', 'sell', 'puts', 'downside', 'short', 'decline', 'downgrade',
            'weak', 'red', 'lower', 'loss', 'losses', 'miss', 'misses', 'missed',
            'negative', 'trouble', 'risk', 'risky', 'risks', 'fall', 'falling',
            'failed', 'failure', 'drop', 'dropping', 'dumping', 'downtrend',
            'selloff', 'sell-off', 'crash', 'crashing', 'bankruptcy', 'concern',
            'concerns', 'worried', 'worry', 'cautious', 'warning', 'disappointing'
        ])
        
        # Intensity modifiers
        self.intensifiers = {
            'very': 1.5,
            'extremely': 2.0,
            'incredibly': 2.0,
            'massively': 2.0,
            'hugely': 1.8,
            'strongly': 1.5,
            'significantly': 1.5,
            'substantially': 1.5,
            'deeply': 1.5,
            'absolutely': 1.8,
            'remarkably': 1.5,
            'exceptionally': 1.7,
            'quite': 1.2,
            'somewhat': 0.8,
            'slightly': 0.5,
            'marginally': 0.3,
            'barely': 0.2
        }
        
        # Negation terms
        self.negations = set([
            'no', 'not', 'none', 'nobody', 'nothing', 'nowhere', 'never', 
            'rarely', 'hardly', 'scarcely', 'barely', 'don\'t', 'doesn\'t', 
            'didn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'haven\'t',
            'hasn\'t', 'hadn\'t', 'won\'t', 'wouldn\'t', 'can\'t', 'cannot',
            'couldn\'t', 'shouldn\'t', 'without'
        ])
    
    def add_positive_terms(self, terms):
        """
        Add terms to positive lexicon
        
        Parameters:
        -----------
        terms : list or set
            Terms to add
        """
        self.positive_terms.update(terms)
    
    def add_negative_terms(self, terms):
        """
        Add terms to negative lexicon
        
        Parameters:
        -----------
        terms : list or set
            Terms to add
        """
        self.negative_terms.update(terms)
    
    def add_intensifiers(self, intensifiers_dict):
        """
        Add intensity modifiers
        
        Parameters:
        -----------
        intensifiers_dict : dict
            Dictionary mapping terms to intensity values
        """
        self.intensifiers.update(intensifiers_dict)
    
    def add_negations(self, terms):
        """
        Add negation terms
        
        Parameters:
        -----------
        terms : list or set
            Terms to add
        """
        self.negations.update(terms)
    
    def analyze_text(self, text, text_processor=None):
        """
        Analyze text using the financial lexicon
        
        Parameters:
        -----------
        text : str
            Text to analyze
        text_processor : StockTextProcessor, optional
            Text processor to use for preprocessing
            
        Returns:
        --------
        dict
            Dictionary with lexicon-based sentiment analysis results
        """
        # Preprocess text if processor provided
        if text_processor is not None:
            text = text_processor.clean_text(text)
        
        # Split into words
        words = text.lower().split()
        
        # Initialize counters
        positive_count = 0
        negative_count = 0
        
        # Initialize variables to track negation and intensifiers
        negated = False
        intensity_modifier = 1.0
        
        # Analyze words
        for i, word in enumerate(words):
            # Check for negations
            if word in self.negations:
                negated = True
                continue
            
            # Check for intensifiers
            if word in self.intensifiers:
                intensity_modifier = self.intensifiers[word]
                continue
            
            # Check sentiment
            if word in self.positive_terms:
                if negated:
                    negative_count += intensity_modifier
                else:
                    positive_count += intensity_modifier
            elif word in self.negative_terms:
                if negated:
                    positive_count += intensity_modifier
                else:
                    negative_count += intensity_modifier
            
            # Reset negation and intensity after using them
            negated = False
            intensity_modifier = 1.0
        
        # Calculate sentiment score (-1 to 1)
        total_count = positive_count + negative_count
        if total_count > 0:
            sentiment_score = (positive_count - negative_count) / total_count
        else:
            sentiment_score = 0
        
        # Determine sentiment category
        if sentiment_score >= 0.2:
            sentiment = 'positive'
        elif sentiment_score <= -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_words': [word for word in words if word in self.positive_terms],
            'negative_words': [word for word in words if word in self.negative_terms]
        }
    
    def batch_analyze(self, texts, text_processor=None):
        """
        Analyze a batch of texts using the financial lexicon
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        text_processor : StockTextProcessor, optional
            Text processor to use for preprocessing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with lexicon-based sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_text(text, text_processor)
            results.append(result)
        return pd.DataFrame(results)
    
    def save_lexicon(self, filepath):
        """
        Save this lexicon to a file
        
        Parameters:
        -----------
        filepath : str
            File path to save lexicon to
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Financial lexicon saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving financial lexicon: {e}")
    
    @classmethod
    def load_lexicon(cls, filepath):
        """
        Load lexicon from file
        
        Parameters:
        -----------
        filepath : str
            File path to load lexicon from
            
        Returns:
        --------
        StockMarketLexicon
            Loaded lexicon
        """
        try:
            with open(filepath, 'rb') as f:
                lexicon = pickle.load(f)
            logger.info(f"Financial lexicon loaded from {filepath}")
            return lexicon
        except Exception as e:
            logger.error(f"Error loading financial lexicon: {e}")
            return cls()  # Return a new instance if loading fails


# If script is run directly, demonstrate basic usage
if __name__ == "__main__":
    # Example text
    example_texts = [
        "I am very bullish on $AAPL, their new products look amazing.",
        "Extremely bearish on $TSLA, the company is facing significant challenges.",
        "Not sure about $AMZN, waiting to see their earnings report."
    ]
    
    # Initialize analyzer and lexicon
    analyzer = BasicSentimentAnalyzer()
    lexicon = StockMarketLexicon()
    
    # Analyze texts using TextBlob
    print("TextBlob Analysis:")
    for text in example_texts:
        result = analyzer.analyze_sentiment_textblob(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}, Polarity: {result['polarity']:.2f}")
        print()
    
    # Analyze texts using lexicon
    print("\nLexicon Analysis:")
    for text in example_texts:
        result = lexicon.analyze_text(text, analyzer.text_processor)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']}, Score: {result['sentiment_score']:.2f}")
        print(f"Positive words: {result['positive_words']}")
        print(f"Negative words: {result['negative_words']}")
        print()