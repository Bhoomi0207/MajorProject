"""
Advanced Emotional Sentiment Analysis for Financial Texts
Implements multiple ML models (Naive Bayes, SVM) and emotional category detection
"""
import numpy as np
import pandas as pd
import logging
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from textblob import TextBlob
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification

from src.preprocessing.text_processor import StockTextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing additional libraries for emotion detection
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        import datasets
        HAS_DATASETS = True
    except ImportError:
        logger.warning("Datasets library not available. Some features will be limited.")
        HAS_DATASETS = False
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("Transformers library not available. Some emotion detection features will be limited.")
    HAS_TRANSFORMERS = False
    HAS_DATASETS = False

# Default emotion categories for stock market sentiment
EMOTION_CATEGORIES = ['fear', 'anger', 'joy', 'optimism', 'pessimism', 'uncertainty', 'surprise']

class EmotionalSentimentAnalyzer:
    """
    Advanced analyzer with classification of sentiment intensity and emotional categories
    Combines multiple models (NB, SVM) for financial sentiment analysis and emotion detection
    """
    
    def __init__(self, text_processor=None, use_gpu=None):
        """
        Initialize the emotional sentiment analyzer
        
        Parameters:
        -----------
        text_processor : StockTextProcessor, optional
            Text processor for preprocessing
        use_gpu : bool, optional
            Whether to use GPU for inference if available
        """
        self.text_processor = text_processor if text_processor is not None else StockTextProcessor()
        
        # Determine device for inference
        if use_gpu is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize VADER for intensity scores
        self.sid = SentimentIntensityAnalyzer()
        
        # Initialize emotion detection model placeholders
        self.emotion_model = None
        self.emotion_tokenizer = None
        self.emotion_pipeline = None
        
        # Emotion lexicons and mappings
        self.emotion_lexicons = self._initialize_emotion_lexicons()
    
    def _initialize_emotion_lexicons(self):
        """
        Initialize lexicons for various emotions
        """
        emotion_lexicons = {
            'fear': [
                'risk', 'threat', 'danger', 'worry', 'concern', 'afraid', 'fear', 'panic', 'anxiety', 'nervous',
                'scared', 'frightened', 'terrified', 'uneasy', 'dread', 'alarm', 'apprehension', 'crisis',
                'emergency', 'exposure', 'vulnerable', 'uncertain', 'collapse', 'crash', 'bankruptcy'
            ],
            'anger': [
                'angry', 'furious', 'outraged', 'upset', 'mad', 'irritated', 'annoyed', 'frustrating', 'disappointing',
                'unfair', 'ridiculous', 'absurd', 'scam', 'corruption', 'mismanagement', 'scandal', 'lawsuit',
                'manipulation', 'exploitation', 'excessive', 'greedy', 'dishonest'
            ],
            'joy': [
                'happy', 'excited', 'pleased', 'delighted', 'thrilled', 'satisfied', 'content', 'confident', 'optimistic',
                'hopeful', 'enthusiastic', 'excellent', 'wonderful', 'amazing', 'fantastic', 'great', 'good', 'impressive',
                'breakthrough', 'triumph', 'success', 'celebration', 'achievement', 'milestone', 'reward'
            ],
            'optimism': [
                'optimistic', 'bullish', 'promising', 'positive', 'potential', 'opportunity', 'growth', 'improvement',
                'recovery', 'progress', 'advance', 'outlook', 'prospect', 'future', 'momentum', 'development',
                'innovation', 'strategy', 'leadership', 'vision', 'expansion', 'upside', 'rally'
            ],
            'pessimism': [
                'pessimistic', 'bearish', 'negative', 'decline', 'downturn', 'recession', 'contraction', 'slowdown',
                'stagnation', 'weak', 'poor', 'difficult', 'challenging', 'problem', 'obstacle', 'barrier', 'limitation',
                'constraint', 'headwind', 'underperform', 'disappointment', 'warning', 'downside', 'struggle', 'doubt'
            ],
            'uncertainty': [
                'uncertain', 'unclear', 'unknown', 'unpredictable', 'volatile', 'unstable', 'variable', 'fluctuating',
                'questionable', 'ambiguous', 'vague', 'complex', 'confusing', 'complicated', 'risky', 'speculative',
                'if', 'may', 'might', 'could', 'possibly', 'perhaps', 'approximately', 'around', 'estimated', 'projected'
            ],
            'surprise': [
                'surprise', 'unexpected', 'shocked', 'astonished', 'amazed', 'startled', 'stunning', 'remarkable',
                'extraordinary', 'unprecedented', 'unusual', 'dramatic', 'sudden', 'sharp', 'significant', 'substantial',
                'exceeded', 'beat', 'missed', 'underestimated', 'overestimated', 'contrary', 'different', 'unexpected', 'shocking'
            ]
        }
        
        return emotion_lexicons
    
    def load_emotion_model(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        """
        Load a pre-trained emotion detection model
        
        Parameters:
        -----------
        model_name : str
            Name of the emotion detection model to use
        """
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers library not available. Using lexicon-based emotion detection instead.")
            return
            
        try:
            logger.info(f"Loading emotion detection model: {model_name}")
            
            # Load tokenizer and model
            self.emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            
            # Create emotion detection pipeline
            self.emotion_pipeline = pipeline(
                "text-classification", 
                model=self.emotion_model, 
                tokenizer=self.emotion_tokenizer,
                top_k=None,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Emotion detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading emotion detection model: {e}")
            logger.info("Falling back to lexicon-based emotion detection")
    
    def train_models(self, texts, labels, dev_texts=None, dev_labels=None, test_size=0.2):
        """
        Train multiple sentiment classification models
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Training texts
        labels : list or pandas.Series
            Sentiment labels for training
        dev_texts : list or pandas.Series, optional
            Development set texts
        dev_labels : list or pandas.Series, optional
            Development set labels
        test_size : float
            Size of test set if dev set not provided
            
        Returns:
        --------
        dict
            Dictionary with model performance information
        """
        # Process texts
        processed_texts = [self.text_processor.preprocess_text(text) for text in texts]
        
        # Split data if dev set not provided
        if dev_texts is None or dev_labels is None:
            train_texts, dev_texts, train_labels, dev_labels = train_test_split(
                processed_texts, labels, test_size=test_size, random_state=42
            )
        else:
            train_texts, train_labels = processed_texts, labels
            dev_texts = [self.text_processor.preprocess_text(text) for text in dev_texts]
        
        # Define models to train
        model_configs = {
            'naive_bayes': {
                'model': MultinomialNB(),
                'vectorizer': TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
            },
            'svm': {
                'model': LinearSVC(C=1.0, class_weight='balanced'),
                'vectorizer': TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
            },
            'logistic_regression': {
                'model': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000),
                'vectorizer': TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
            }
        }
        
        # Train all models
        results = {}
        for model_name, config in model_configs.items():
            try:
                logger.info(f"Training {model_name} model")
                
                # Create and train pipeline
                pipeline = Pipeline([
                    ('vectorizer', config['vectorizer']),
                    ('model', config['model'])
                ])
                
                pipeline.fit(train_texts, train_labels)
                
                # Evaluate on dev set
                dev_pred = pipeline.predict(dev_texts)
                accuracy = accuracy_score(dev_labels, dev_pred)
                f1 = f1_score(dev_labels, dev_pred, average='weighted')
                report = classification_report(dev_labels, dev_pred, output_dict=True)
                
                logger.info(f"{model_name} trained. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                # Store model and metrics
                self.models[model_name] = {
                    'pipeline': pipeline,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'report': report
                }
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'report': report
                }
                
            except Exception as e:
                logger.error(f"Error training {model_name} model: {e}")
        
        # Train ensemble model if multiple models available
        if len(self.models) >= 2:
            try:
                logger.info("Training ensemble model")
                
                # Create voting classifier
                estimators = [
                    (name, model_info['pipeline']) 
                    for name, model_info in self.models.items()
                ]
                
                # Create a new TF-IDF vectorizer for the ensemble
                ensemble_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
                X_train_tfidf = ensemble_vectorizer.fit_transform(train_texts)
                X_dev_tfidf = ensemble_vectorizer.transform(dev_texts)
                
                # Train individual models for the ensemble using the same vectorization
                fitted_models = []
                for model_name, config in model_configs.items():
                    if model_name in self.models:
                        model = config['model']
                        model.fit(X_train_tfidf, train_labels)
                        fitted_models.append((model_name, model))
                
                # Create and train voting classifier
                voting_clf = VotingClassifier(
                    estimators=fitted_models,
                    voting='hard'  # Use 'soft' for probability-based voting if all models support predict_proba
                )
                
                voting_clf.fit(X_train_tfidf, train_labels)
                
                # Evaluate ensemble
                ensemble_pred = voting_clf.predict(X_dev_tfidf)
                ensemble_accuracy = accuracy_score(dev_labels, ensemble_pred)
                ensemble_f1 = f1_score(dev_labels, ensemble_pred, average='weighted')
                ensemble_report = classification_report(dev_labels, ensemble_pred, output_dict=True)
                
                logger.info(f"Ensemble model trained. Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
                
                # Store ensemble model
                self.models['ensemble'] = {
                    'pipeline': Pipeline([
                        ('vectorizer', ensemble_vectorizer),
                        ('model', voting_clf)
                    ]),
                    'accuracy': ensemble_accuracy,
                    'f1_score': ensemble_f1,
                    'report': ensemble_report
                }
                
                results['ensemble'] = {
                    'accuracy': ensemble_accuracy,
                    'f1_score': ensemble_f1,
                    'report': ensemble_report
                }
                
            except Exception as e:
                logger.error(f"Error training ensemble model: {e}")
        
        return results
    
    def analyze_sentiment(self, text, model_name='best'):
        """
        Perform comprehensive sentiment analysis with intensity and emotions
        
        Parameters:
        -----------
        text : str
            Text to analyze
        model_name : str
            Model to use ('naive_bayes', 'svm', 'ensemble', or 'best')
            
        Returns:
        --------
        dict
            Dictionary with comprehensive sentiment analysis results
        """
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(text)
        
        # Initialize results dictionary
        result = {
            'original_text': text,
            'processed_text': processed_text
        }
        
        # Get sentiment intensity using VADER
        vader_scores = self.sid.polarity_scores(processed_text)
        result.update({
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu']
        })
        
        # Get TextBlob scores
        blob = TextBlob(processed_text)
        result.update({
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        })
        
        # Determine sentiment label and intensity
        # Combine VADER and TextBlob for a more accurate assessment
        compound_score = vader_scores['compound']
        tb_polarity = blob.sentiment.polarity
        
        # Weighted average for combined sentiment score
        combined_score = (compound_score * 0.7) + (tb_polarity * 0.3)
        result['sentiment_score'] = combined_score
        
        # Determine intensity and label
        if combined_score >= 0.5:
            sentiment_label = 'very_positive'
            intensity = 'strong'
        elif combined_score >= 0.2:
            sentiment_label = 'positive'
            intensity = 'moderate'
        elif combined_score > -0.2:
            sentiment_label = 'neutral'
            intensity = 'weak'
        elif combined_score > -0.5:
            sentiment_label = 'negative'
            intensity = 'moderate'
        else:
            sentiment_label = 'very_negative'
            intensity = 'strong'
            
        result.update({
            'sentiment_label': sentiment_label,
            'sentiment_intensity': intensity
        })
        
        # Apply ML models if available
        if self.models:
            if model_name == 'best':
                # Choose the best model based on accuracy
                best_model = max(self.models.items(), key=lambda x: x[1]['accuracy'])
                model_name = best_model[0]
                
            if model_name in self.models:
                model_info = self.models[model_name]
                pipeline = model_info['pipeline']
                
                # Get prediction
                ml_label = pipeline.predict([processed_text])[0]
                
                # Get confidence if available
                if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                    probabilities = pipeline.predict_proba([processed_text])[0]
                    confidence = np.max(probabilities)
                else:
                    # For models without predict_proba like LinearSVC
                    confidence = None
                
                result.update({
                    'ml_sentiment': ml_label,
                    'ml_confidence': confidence,
                    'ml_model': model_name
                })
        
        # Detect emotions using lexicons
        emotions = self._detect_emotions_lexicon(processed_text)
        result['emotions'] = emotions
        
        # Add dominant emotion
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            result['dominant_emotion'] = dominant_emotion[0]
            result['dominant_emotion_score'] = dominant_emotion[1]
        
        # Try transformer-based emotion detection if available
        if self.emotion_pipeline:
            try:
                transformer_emotions = self._detect_emotions_transformer(text)
                result['transformer_emotions'] = transformer_emotions
            except Exception as e:
                logger.error(f"Error in transformer emotion detection: {e}")
        
        return result
    
    def _detect_emotions_lexicon(self, text):
        """
        Detect emotions in text using lexicon-based approach
        
        Parameters:
        -----------
        text : str
            Preprocessed text to analyze
            
        Returns:
        --------
        dict
            Dictionary of emotion scores
        """
        words = text.lower().split()
        emotion_scores = defaultdict(float)
        
        # Count emotion words
        for emotion, lexicon in self.emotion_lexicons.items():
            count = 0
            for word in words:
                if word in lexicon:
                    count += 1
            
            # Normalize by text length
            if words:
                emotion_scores[emotion] = count / len(words)
        
        return dict(emotion_scores)
    
    def _detect_emotions_transformer(self, text):
        """
        Detect emotions using transformer model
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary of emotion scores
        """
        if not self.emotion_pipeline:
            return {}
            
        try:
            result = self.emotion_pipeline(text)[0]
            
            # Format results as dictionary
            emotions = {item['label']: item['score'] for item in result}
            return emotions
            
        except Exception as e:
            logger.error(f"Error in transformer emotion detection: {e}")
            return {}
    
    def batch_analyze(self, texts, model_name='best', batch_size=32, show_progress=True):
        """
        Analyze sentiment for a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        model_name : str
            Model to use for sentiment classification
        batch_size : int
            Size of batches for processing
        show_progress : bool
            Whether to show a progress bar
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment analysis results
        """
        # Convert to list if Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        results = []
        
        # Process in batches to avoid memory issues
        try:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        except ImportError:
            iterator = range(0, len(texts), batch_size)
            
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Analyze each text in batch
            for text in batch_texts:
                try:
                    result = self.analyze_sentiment(text, model_name)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing text: {e}")
                    # Add basic result for failed analysis
                    results.append({
                        'original_text': text,
                        'processed_text': text,
                        'vader_compound': 0,
                        'sentiment_label': 'neutral',
                        'sentiment_intensity': 'weak',
                        'emotions': {},
                        'error': str(e)
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def analyze_dataframe(self, df, text_column, result_prefix='sentiment_', model_name='best'):
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
        model_name : str
            Model to use for sentiment classification
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sentiment analysis columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Analyze sentiment for each text
        sentiments = self.batch_analyze(df[text_column], model_name)
        
        # Add basic sentiment columns
        result_df[f'{result_prefix}label'] = sentiments['sentiment_label']
        result_df[f'{result_prefix}score'] = sentiments['sentiment_score']
        result_df[f'{result_prefix}intensity'] = sentiments['sentiment_intensity']
        
        # Add VADER scores
        result_df[f'{result_prefix}vader_compound'] = sentiments['vader_compound']
        
        # Add TextBlob scores
        result_df[f'{result_prefix}textblob_polarity'] = sentiments['textblob_polarity']
        result_df[f'{result_prefix}textblob_subjectivity'] = sentiments['textblob_subjectivity']
        
        # Add ML model results if available
        if 'ml_sentiment' in sentiments.columns:
            result_df[f'{result_prefix}ml_sentiment'] = sentiments['ml_sentiment']
            if 'ml_confidence' in sentiments.columns:
                result_df[f'{result_prefix}ml_confidence'] = sentiments['ml_confidence']
            result_df[f'{result_prefix}ml_model'] = sentiments['ml_model']
        
        # Add dominant emotion
        if 'dominant_emotion' in sentiments.columns:
            result_df[f'{result_prefix}dominant_emotion'] = sentiments['dominant_emotion']
            result_df[f'{result_prefix}dominant_emotion_score'] = sentiments['dominant_emotion_score']
        
        # Extract emotions to separate columns
        for emotion in EMOTION_CATEGORIES:
            result_df[f'{result_prefix}emotion_{emotion}'] = sentiments.apply(
                lambda x: x['emotions'].get(emotion, 0) if isinstance(x['emotions'], dict) else 0, 
                axis=1
            )
        
        return result_df
    
    def plot_sentiment_distribution(self, results_df, save_path=None):
        """
        Plot sentiment distribution from analysis results
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with sentiment analysis results
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Count sentiment labels
        sentiment_counts = results_df['sentiment_label'].value_counts()
        
        # Create bar chart with custom colors
        colors = {
            'very_positive': '#1E8F4E',  # Dark green
            'positive': '#7ED957',      # Light green
            'neutral': '#DBDBDB',       # Gray
            'negative': '#FF6B6B',      # Light red
            'very_negative': '#D62828'  # Dark red
        }
        
        # Ensure all categories exist
        for label in colors.keys():
            if label not in sentiment_counts:
                sentiment_counts[label] = 0
        
        # Sort by sentiment intensity
        sentiment_counts = sentiment_counts.reindex(['very_positive', 'positive', 'neutral', 'negative', 'very_negative'])
        
        # Create bar chart
        ax = sentiment_counts.plot(kind='bar', color=[colors[label] for label in sentiment_counts.index])
        
        # Add count labels on bars
        for i, count in enumerate(sentiment_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_emotion_distribution(self, results_df, save_path=None):
        """
        Plot emotion distribution from analysis results
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with sentiment analysis results
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Count dominant emotions
        if 'dominant_emotion' in results_df.columns:
            emotion_counts = results_df['dominant_emotion'].value_counts()
            
            # Create bar chart with colorful palette
            ax = emotion_counts.plot(kind='bar', colormap='viridis')
            
            # Add count labels on bars
            for i, count in enumerate(emotion_counts):
                plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
            
            plt.title('Dominant Emotion Distribution')
            plt.xlabel('Emotion Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            
            plt.show()
            
        else:
            # If no dominant emotion, use emotion scores
            emotion_scores = {}
            
            for emotion in EMOTION_CATEGORIES:
                col_name = f'emotion_{emotion}'
                if col_name in results_df.columns:
                    emotion_scores[emotion] = results_df[col_name].mean()
            
            if emotion_scores:
                # Convert to Series and plot
                emotion_series = pd.Series(emotion_scores)
                ax = emotion_series.plot(kind='bar', colormap='viridis')
                
                plt.title('Average Emotion Scores')
                plt.xlabel('Emotion Category')
                plt.ylabel('Average Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save if path provided
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path)
                
                plt.show()
    
    def save_analyzer(self, filepath):
        """
        Save the analyzer to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the analyzer
        """
        try:
            # Create a copy without non-picklable objects
            save_obj = EmotionalSentimentAnalyzer(self.text_processor)
            save_obj.models = self.models
            save_obj.emotion_lexicons = self.emotion_lexicons
            
            # Don't save transformer models - they should be reloaded
            save_obj.emotion_model = None
            save_obj.emotion_tokenizer = None
            save_obj.emotion_pipeline = None
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_obj, f)
                
            logger.info(f"Analyzer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving analyzer: {e}")
    
    @classmethod
    def load_analyzer(cls, filepath, load_emotion_model=True):
        """
        Load an analyzer from a file
        
        Parameters:
        -----------
        filepath : str
            Path to load the analyzer from
        load_emotion_model : bool
            Whether to load the emotion transformer model
            
        Returns:
        --------
        EmotionalSentimentAnalyzer
            Loaded analyzer
        """
        try:
            with open(filepath, 'rb') as f:
                analyzer = pickle.load(f)
                
            # Initialize VADER
            analyzer.sid = SentimentIntensityAnalyzer()
            
            # Load emotion model if requested
            if load_emotion_model:
                analyzer.load_emotion_model()
                
            logger.info(f"Analyzer loaded from {filepath}")
            return analyzer
            
        except Exception as e:
            logger.error(f"Error loading analyzer: {e}")
            return cls()  # Return a new instance if loading fails


# Example usage
if __name__ == "__main__":
    # Example texts
    example_texts = [
        "I am very bullish on $AAPL, their new products look amazing and will definitely boost their revenue!",
        "The market is looking extremely risky, I'm worried about a potential crash in the next few weeks.",
        "Not sure about $AMZN, their growth is slowing but they still dominate e-commerce.",
        "I'm outraged by the SEC's latest decision, this will hurt small investors and benefit only Wall Street!",
        "The company's latest earnings report exceeded all expectations, I'm very excited about their future!"
    ]
    
    # Initialize analyzer
    analyzer = EmotionalSentimentAnalyzer()
    
    # Try to load emotion detection model
    try:
        analyzer.load_emotion_model()
    except Exception as e:
        print(f"Could not load emotion model: {e}")
    
    # Analyze texts
    print("Emotional Sentiment Analysis Results:")
    for text in example_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment_label']} (Score: {result['sentiment_score']:.2f}, Intensity: {result['sentiment_intensity']})")
        print(f"VADER compound: {result['vader_compound']:.2f}")
        print(f"TextBlob polarity: {result['textblob_polarity']:.2f}, subjectivity: {result['textblob_subjectivity']:.2f}")
        
        print("Emotions:")
        for emotion, score in result['emotions'].items():
            if score > 0:
                print(f"  - {emotion}: {score:.3f}")
        
        if 'dominant_emotion' in result:
            print(f"Dominant emotion: {result['dominant_emotion']} ({result['dominant_emotion_score']:.3f})")
        
        print("-" * 80)