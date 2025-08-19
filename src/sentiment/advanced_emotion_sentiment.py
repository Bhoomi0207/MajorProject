"""
Enhanced Emotional Sentiment Analysis for Financial Texts
Extends the EmotionalSentimentAnalyzer with more advanced features:
- More granular sentiment intensity scale (7-point)
- Enhanced emotional categories and fine-grained classification
- Financial market-specific emotion interpretation
- Advanced visualization capabilities
- Contextual sentiment analysis based on market conditions
"""
import numpy as np
import pandas as pd
import logging
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import torch
import math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from textblob import TextBlob
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing.text_processor import StockTextProcessor
from src.sentiment.emotion_sentiment import EmotionalSentimentAnalyzer, EMOTION_CATEGORIES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing additional libraries for advanced NLP
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    HAS_ADVANCED_NLP = True
except ImportError:
    logger.warning("Some advanced NLP libraries not available. Some features will be limited.")
    HAS_ADVANCED_NLP = False

# Extended emotion categories adding more fine-grained emotions
EXTENDED_EMOTION_CATEGORIES = {
    'fear': ['anxiety', 'worry', 'nervousness', 'dread', 'panic', 'terror', 'apprehension'],
    'anger': ['frustration', 'irritation', 'outrage', 'annoyance', 'fury', 'indignation', 'resentment'],
    'joy': ['happiness', 'excitement', 'delight', 'pleasure', 'contentment', 'satisfaction', 'elation'],
    'optimism': ['hopefulness', 'confidence', 'positivity', 'enthusiasm', 'conviction', 'assurance', 'bullishness'],
    'pessimism': ['negativity', 'bearishness', 'hopelessness', 'cynicism', 'despair', 'defeatism', 'doubt'],
    'uncertainty': ['confusion', 'ambiguity', 'doubt', 'hesitation', 'perplexity', 'indecision', 'skepticism'],
    'surprise': ['shock', 'amazement', 'astonishment', 'bewilderment', 'disbelief', 'awe', 'wonder'],
    'trust': ['belief', 'faith', 'credibility', 'reliability', 'dependability', 'conviction', 'confidence'],
    'anticipation': ['expectation', 'prediction', 'projection', 'forecast', 'planning', 'preparation', 'foresight']
}

# Market-specific emotion interpretations
MARKET_EMOTION_INTERPRETATIONS = {
    'fear': "Risk aversion, potential sell-off, defensive positioning",
    'anger': "Reaction to missed expectations, regulatory concerns, or corporate governance issues",
    'joy': "Positive reaction to earnings beats, product launches, or market rallies",
    'optimism': "Bullish outlook, confidence in future performance, buying opportunity signals",
    'pessimism': "Bearish outlook, expectation of downside, negative sentiment toward prospects",
    'uncertainty': "Volatility expectation, hedging activity, wait-and-see approach",
    'surprise': "Reaction to unexpected news, earnings surprises, or market shocks",
    'trust': "Confidence in management, strategy, or fundamentals",
    'anticipation': "Position-taking ahead of events, preparation for expected catalysts"
}

# 7-point sentiment intensity scale
SENTIMENT_INTENSITY_SCALE = {
    'extremely_positive': (0.75, 1.0),
    'very_positive': (0.5, 0.75),
    'positive': (0.2, 0.5),
    'neutral': (-0.2, 0.2),
    'negative': (-0.5, -0.2),
    'very_negative': (-0.75, -0.5),
    'extremely_negative': (-1.0, -0.75)
}

# Intensity scale colors
INTENSITY_COLORS = {
    'extremely_positive': '#006400',  # Dark green
    'very_positive': '#228B22',       # Forest green
    'positive': '#7ED957',            # Light green
    'neutral': '#DBDBDB',             # Gray
    'negative': '#FF9999',            # Light red
    'very_negative': '#FF6B6B',       # Darker red
    'extremely_negative': '#D62828'   # Darkest red
}

class EnhancedEmotionalAnalyzer(EmotionalSentimentAnalyzer):
    """
    Enhanced emotional sentiment analyzer with more granular analysis, 
    financial market interpretation, and advanced visualizations
    """
    
    def __init__(self, text_processor=None, use_gpu=None, market_context=None):
        """
        Initialize enhanced emotional analyzer
        
        Parameters:
        -----------
        text_processor : TextProcessor, optional
            Text processor for preprocessing
        use_gpu : bool, optional
            Whether to use GPU for inference if available
        market_context : dict, optional
            Market context information for contextual analysis
        """
        # Initialize the base class
        super().__init__(text_processor, use_gpu)
        
        # Enhanced emotion lexicons with fine-grained emotions
        self.extended_emotion_lexicons = self._initialize_extended_emotion_lexicons()
        
        # Market context for contextual analysis
        self.market_context = market_context or {}
        
        # Initialize advanced NLP components if available
        self.spacy_nlp = None
        if HAS_ADVANCED_NLP:
            try:
                import spacy
                self.spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy NLP model for enhanced text analysis")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
        
        # Initialize advanced model components
        self.advanced_models = {}
        self.fine_emotion_model = None
        self.fine_emotion_tokenizer = None
        self.fine_emotion_pipeline = None
    
    def _initialize_extended_emotion_lexicons(self):
        """
        Initialize extended emotion lexicons with more fine-grained emotions
        """
        # Start with the basic emotion lexicons from the parent class
        extended_lexicons = {}
        
        # Build lexicons for each emotion category and its sub-emotions
        for emotion, sub_emotions in EXTENDED_EMOTION_CATEGORIES.items():
            # Get the base lexicon if it exists in the parent class
            base_words = self.emotion_lexicons.get(emotion, [])
            
            # Create a new set for this emotion and its sub-emotions
            emotion_words = set(base_words)
            
            # Add typical words for each sub-emotion
            for sub_emotion in sub_emotions:
                # Add the sub-emotion itself and common variants
                emotion_words.add(sub_emotion)
                
                # Add related words based on common patterns
                if sub_emotion.endswith('y'):
                    emotion_words.add(sub_emotion[:-1] + 'ied')
                    emotion_words.add(sub_emotion[:-1] + 'ies')
                elif sub_emotion.endswith('ce'):
                    emotion_words.add(sub_emotion[:-2] + 'ing')
                else:
                    emotion_words.add(sub_emotion + 'ing')
                    emotion_words.add(sub_emotion + 'ed')
            
            extended_lexicons[emotion] = list(emotion_words)
            
            # Also create lexicons for each sub-emotion
            for sub_emotion in sub_emotions:
                sub_words = [word for word in emotion_words 
                            if sub_emotion in word or word in sub_emotion]
                sub_words.append(sub_emotion)
                extended_lexicons[sub_emotion] = sub_words
        
        # Add financial market-specific emotion words
        extended_lexicons['bullish'] = [
            'bullish', 'bull', 'long', 'upside', 'uptrend', 'buy', 'calls', 
            'rally', 'breakout', 'accumulate', 'moon', 'rip', 'growth', 'winner'
        ]
        
        extended_lexicons['bearish'] = [
            'bearish', 'bear', 'short', 'downside', 'downtrend', 'sell', 'puts',
            'correction', 'crash', 'tank', 'dump', 'collapse', 'fall', 'drop'
        ]
        
        extended_lexicons['fomo'] = [
            'fomo', 'fear of missing out', 'missing out', 'jumped in', 'late',
            'bandwagon', 'trend', 'everyone', 'popularity', 'viral', 'hot'
        ]
        
        extended_lexicons['fud'] = [
            'fud', 'fear uncertainty doubt', 'manipulation', 'conspiracy',
            'hit piece', 'attack', 'spreaders', 'shorts', 'bears', 'coordinated'
        ]
        
        return extended_lexicons
    
    def load_fine_emotion_model(self, model_name="SamLowe/roberta-base-go_emotions"):
        """
        Load fine-grained emotion detection model
        
        Parameters:
        -----------
        model_name : str
            Name of the fine-grained emotion model to use
        """
        if not HAS_ADVANCED_NLP:
            logger.warning("Advanced NLP libraries not available. Using lexicon-based emotion detection instead.")
            return
            
        try:
            logger.info(f"Loading fine-grained emotion detection model: {model_name}")
            
            # Load tokenizer and model
            self.fine_emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.fine_emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            
            # Create emotion detection pipeline
            self.fine_emotion_pipeline = pipeline(
                "text-classification", 
                model=self.fine_emotion_model, 
                tokenizer=self.fine_emotion_tokenizer,
                top_k=5,  # Return top 5 emotions
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Fine-grained emotion detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-grained emotion model: {e}")
            logger.info("Falling back to regular emotion detection")
    
    def train_advanced_models(self, texts, labels, dev_texts=None, dev_labels=None, test_size=0.2,
                             grid_search=False):
        """
        Train advanced sentiment classification models with hyperparameter tuning
        
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
        grid_search : bool
            Whether to perform grid search for hyperparameter tuning
            
        Returns:
        --------
        dict
            Dictionary with model performance information
        """
        # Process texts using spaCy if available for better feature extraction
        if self.spacy_nlp:
            logger.info("Using spaCy for advanced text processing")
            processed_texts = []
            for text in texts:
                # Basic preprocessing
                cleaned_text = self.text_processor.preprocess_text(text)
                # Advanced NLP processing
                doc = self.spacy_nlp(cleaned_text)
                # Extract lemmas for non-stop words
                lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                processed_texts.append(" ".join(lemmas))
        else:
            # Fall back to base class preprocessing
            processed_texts = [self.text_processor.preprocess_text(text) for text in texts]
        
        # Split data if dev set not provided
        if dev_texts is None or dev_labels is None:
            train_texts, dev_texts, train_labels, dev_labels = train_test_split(
                processed_texts, labels, test_size=test_size, random_state=42
            )
        else:
            train_texts, train_labels = processed_texts, labels
            # Process dev texts
            if self.spacy_nlp:
                processed_dev = []
                for text in dev_texts:
                    cleaned_text = self.text_processor.preprocess_text(text)
                    doc = self.spacy_nlp(cleaned_text)
                    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                    processed_dev.append(" ".join(lemmas))
                dev_texts = processed_dev
            else:
                dev_texts = [self.text_processor.preprocess_text(text) for text in dev_texts]
        
        # Define advanced models with more sophisticated hyperparameters
        model_configs = {
            'advanced_naive_bayes': {
                'model': MultinomialNB(),
                'vectorizer': TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
            },
            'advanced_svm': {
                'model': LinearSVC(C=1.0, class_weight='balanced', dual=False),
                'vectorizer': TfidfVectorizer(max_features=20000, ngram_range=(1, 3))
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
                'vectorizer': TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                'vectorizer': TfidfVectorizer(max_features=15000, ngram_range=(1, 3))
            }
        }
        
        # Hyperparameter grids for grid search
        param_grids = {
            'advanced_naive_bayes': {
                'vectorizer__max_features': [10000, 15000, 20000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'model__alpha': [0.1, 0.5, 1.0]
            },
            'advanced_svm': {
                'vectorizer__max_features': [15000, 20000, 25000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'model__C': [0.1, 1.0, 10.0]
            },
            'random_forest': {
                'vectorizer__max_features': [10000, 15000, 20000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 10, 20, 30]
            },
            'gradient_boosting': {
                'vectorizer__max_features': [10000, 15000, 20000],
                'vectorizer__ngram_range': [(1, 2), (1, 3)],
                'model__n_estimators': [50, 100, 150],
                'model__learning_rate': [0.05, 0.1, 0.2]
            }
        }
        
        # Train all models
        results = {}
        for model_name, config in model_configs.items():
            try:
                logger.info(f"Training {model_name} model")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('vectorizer', config['vectorizer']),
                    ('model', config['model'])
                ])
                
                # Perform grid search if requested
                if grid_search:
                    logger.info(f"Performing grid search for {model_name}")
                    grid = GridSearchCV(
                        pipeline, param_grids[model_name],
                        cv=5, scoring='f1_weighted', n_jobs=-1
                    )
                    grid.fit(train_texts, train_labels)
                    best_pipeline = grid.best_estimator_
                    logger.info(f"Best parameters for {model_name}: {grid.best_params_}")
                else:
                    # Otherwise, just fit the pipeline with default parameters
                    best_pipeline = pipeline
                    best_pipeline.fit(train_texts, train_labels)
                
                # Evaluate on dev set
                dev_pred = best_pipeline.predict(dev_texts)
                accuracy = accuracy_score(dev_labels, dev_pred)
                f1 = f1_score(dev_labels, dev_pred, average='weighted')
                report = classification_report(dev_labels, dev_pred, output_dict=True)
                
                logger.info(f"{model_name} trained. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                # Store model and metrics
                self.advanced_models[model_name] = {
                    'pipeline': best_pipeline,
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
        
        # Train stacked ensemble if multiple models available
        if len(self.advanced_models) >= 2:
            try:
                logger.info("Training stacked ensemble model")
                
                # Create ensemble with all trained models
                estimators = [
                    (name, model_info['pipeline']) 
                    for name, model_info in self.advanced_models.items()
                ]
                
                # Create a voting classifier
                voting_clf = VotingClassifier(
                    estimators=estimators,
                    voting='hard'  # Use 'hard' voting for simplicity
                )
                
                # Create a new TF-IDF vectorizer for the ensemble
                ensemble_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 3))
                ensemble_pipeline = Pipeline([
                    ('vectorizer', ensemble_vectorizer),
                    ('model', voting_clf)
                ])
                
                # Fit the ensemble pipeline
                ensemble_pipeline.fit(train_texts, train_labels)
                
                # Evaluate ensemble
                ensemble_pred = ensemble_pipeline.predict(dev_texts)
                ensemble_accuracy = accuracy_score(dev_labels, ensemble_pred)
                ensemble_f1 = f1_score(dev_labels, ensemble_pred, average='weighted')
                ensemble_report = classification_report(dev_labels, ensemble_pred, output_dict=True)
                
                logger.info(f"Stacked ensemble trained. Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
                
                # Store ensemble model
                self.advanced_models['stacked_ensemble'] = {
                    'pipeline': ensemble_pipeline,
                    'accuracy': ensemble_accuracy,
                    'f1_score': ensemble_f1,
                    'report': ensemble_report
                }
                
                results['stacked_ensemble'] = {
                    'accuracy': ensemble_accuracy,
                    'f1_score': ensemble_f1,
                    'report': ensemble_report
                }
                
            except Exception as e:
                logger.error(f"Error training stacked ensemble model: {e}")
        
        return results
    
    def analyze_sentiment_advanced(self, text, model_name='best', include_fine_emotions=True,
                                  with_market_context=True):
        """
        Perform enhanced sentiment analysis with fine-grained emotions
        
        Parameters:
        -----------
        text : str
            Text to analyze
        model_name : str
            Model to use ('best', or specific model name)
        include_fine_emotions : bool
            Whether to include fine-grained emotion analysis
        with_market_context : bool
            Whether to include market context in analysis
            
        Returns:
        --------
        dict
            Dictionary with comprehensive sentiment analysis results
        """
        # Get base sentiment analysis from parent class
        base_result = self.analyze_sentiment(text, model_name)
        
        # Extend with 7-point intensity scale
        compound_score = base_result['vader_compound']
        tb_polarity = base_result['textblob_polarity']
        
        # Weighted combined score (same as parent class)
        combined_score = (compound_score * 0.7) + (tb_polarity * 0.3)
        
        # Determine 7-point intensity scale
        for intensity_label, (min_val, max_val) in SENTIMENT_INTENSITY_SCALE.items():
            if min_val <= combined_score < max_val:
                enhanced_intensity = intensity_label
                break
        else:
            # Fallback in case we're outside the ranges
            if combined_score >= 1.0:
                enhanced_intensity = 'extremely_positive'
            else:
                enhanced_intensity = 'extremely_negative'
        
        # Update the result with enhanced intensity
        base_result['enhanced_intensity'] = enhanced_intensity
        base_result['intensity_level'] = list(SENTIMENT_INTENSITY_SCALE.keys()).index(enhanced_intensity)
        
        # Detect fine-grained emotions if requested
        if include_fine_emotions:
            # Get fine-grained emotions using the extended lexicons
            fine_emotions = self._detect_fine_emotions(text)
            base_result['fine_emotions'] = fine_emotions
            
            # Try using the transformer-based fine emotion model if available
            if self.fine_emotion_pipeline:
                try:
                    transformer_fine_emotions = self._detect_fine_emotions_transformer(text)
                    base_result['transformer_fine_emotions'] = transformer_fine_emotions
                except Exception as e:
                    logger.error(f"Error in transformer fine emotion detection: {e}")
        
        # Add market context interpretation if requested and context is available
        if with_market_context and self.market_context:
            base_result['market_context'] = self._apply_market_context(base_result)
        
        # Add financial market interpretation for detected emotions
        if 'dominant_emotion' in base_result:
            dominant_emotion = base_result['dominant_emotion']
            if dominant_emotion in MARKET_EMOTION_INTERPRETATIONS:
                base_result['market_interpretation'] = MARKET_EMOTION_INTERPRETATIONS[dominant_emotion]
        
        # Add advanced model prediction if available
        if self.advanced_models:
            if model_name == 'best':
                # Choose best model based on accuracy
                best_model = max(self.advanced_models.items(), key=lambda x: x[1]['accuracy'])
                advanced_model_name = best_model[0]
            elif model_name in self.advanced_models:
                advanced_model_name = model_name
            else:
                # Default to first available model
                advanced_model_name = list(self.advanced_models.keys())[0]
            
            try:
                model_info = self.advanced_models[advanced_model_name]
                pipeline = model_info['pipeline']
                
                # Preprocess text for advanced model
                if self.spacy_nlp:
                    # Use spaCy for advanced preprocessing
                    cleaned_text = self.text_processor.preprocess_text(text)
                    doc = self.spacy_nlp(cleaned_text)
                    # Extract lemmas for non-stop words
                    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                    processed_text = " ".join(lemmas)
                else:
                    # Fall back to regular preprocessing
                    processed_text = self.text_processor.preprocess_text(text)
                
                # Get prediction
                adv_label = pipeline.predict([processed_text])[0]
                
                # Get confidence if available
                if hasattr(pipeline.named_steps['model'], 'predict_proba'):
                    probabilities = pipeline.predict_proba([processed_text])[0]
                    confidence = np.max(probabilities)
                else:
                    confidence = None
                
                base_result.update({
                    'advanced_sentiment': adv_label,
                    'advanced_confidence': confidence,
                    'advanced_model': advanced_model_name
                })
            except Exception as e:
                logger.error(f"Error applying advanced model: {e}")
        
        return base_result
    
    def _detect_fine_emotions(self, text):
        """
        Detect fine-grained emotions using extended lexicons
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary of fine-grained emotion scores
        """
        processed_text = self.text_processor.preprocess_text(text)
        words = processed_text.lower().split()
        
        # Count emotion words for each fine-grained category
        fine_emotion_scores = defaultdict(float)
        
        # First, check the standard emotion categories
        for emotion, lexicon in self.extended_emotion_lexicons.items():
            count = 0
            for word in words:
                if word in lexicon:
                    count += 1
            
            # Normalize by text length
            if words:
                fine_emotion_scores[emotion] = count / len(words)
        
        # Get top emotions (scores > 0)
        top_emotions = {e: s for e, s in fine_emotion_scores.items() if s > 0}
        
        return dict(top_emotions)
    
    def _detect_fine_emotions_transformer(self, text):
        """
        Detect fine-grained emotions using transformer model
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary of fine-grained emotion scores
        """
        if not self.fine_emotion_pipeline:
            return {}
            
        try:
            result = self.fine_emotion_pipeline(text)
            
            # Format results as dictionary of top emotions
            emotions = {item['label']: item['score'] for item in result}
            return emotions
            
        except Exception as e:
            logger.error(f"Error in transformer fine emotion detection: {e}")
            return {}
    
    def _apply_market_context(self, result):
        """
        Apply market context to sentiment analysis
        
        Parameters:
        -----------
        result : dict
            Sentiment analysis result
            
        Returns:
        --------
        dict
            Market context interpretation
        """
        # Start with basic context
        context_result = {
            'applied_context': True
        }
        
        # Extract sentiment and emotion information
        sentiment_label = result.get('sentiment_label', 'neutral')
        sentiment_score = result.get('sentiment_score', 0)
        dominant_emotion = result.get('dominant_emotion', 'uncertainty')
        
        # Apply market trend context if available
        if 'market_trend' in self.market_context:
            trend = self.market_context['market_trend']
            # Check for sentiment-trend alignment
            if (trend == 'bullish' and sentiment_score > 0.2) or (trend == 'bearish' and sentiment_score < -0.2):
                context_result['trend_alignment'] = 'aligned'
                context_result['confidence_multiplier'] = 1.2  # Boost confidence
            else:
                context_result['trend_alignment'] = 'contrarian'
                context_result['confidence_multiplier'] = 0.8  # Reduce confidence
        
        # Apply sector context if available
        if 'sector_performance' in self.market_context and 'sector' in self.market_context:
            sector = self.market_context['sector']
            sector_perf = self.market_context['sector_performance']
            
            context_result['sector_context'] = f"Sector ({sector}) is {sector_perf}"
            
            # Adjust interpretation based on sector performance
            if sector_perf == 'outperforming' and sentiment_score > 0:
                context_result['sector_alignment'] = 'sector strength reinforces positive sentiment'
            elif sector_perf == 'underperforming' and sentiment_score < 0:
                context_result['sector_alignment'] = 'sector weakness reinforces negative sentiment'
        
        # Apply recent news sentiment if available
        if 'recent_news_sentiment' in self.market_context:
            news_sentiment = self.market_context['recent_news_sentiment']
            context_result['news_context'] = f"Recent news sentiment: {news_sentiment}"
            
            # Check for sentiment-news alignment
            if (news_sentiment > 0 and sentiment_score > 0) or (news_sentiment < 0 and sentiment_score < 0):
                context_result['news_alignment'] = 'aligned with recent news'
            else:
                context_result['news_alignment'] = 'contrarian to recent news'
        
        # Generate overall market context interpretation
        if dominant_emotion in MARKET_EMOTION_INTERPRETATIONS:
            interpretation = MARKET_EMOTION_INTERPRETATIONS[dominant_emotion]
            
            # Add trend context
            if 'trend_alignment' in context_result:
                if context_result['trend_alignment'] == 'aligned':
                    interpretation += f" (aligned with {self.market_context.get('market_trend', 'current')} market trend)"
                else:
                    interpretation += f" (contrarian to {self.market_context.get('market_trend', 'current')} market trend)"
            
            context_result['interpretation'] = interpretation
        
        return context_result
    
    def batch_analyze_advanced(self, texts, model_name='best', batch_size=32, 
                             include_fine_emotions=True, with_market_context=True,
                             show_progress=True):
        """
        Perform advanced analysis on a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        model_name : str
            Model to use
        batch_size : int
            Size of batches for processing
        include_fine_emotions : bool
            Whether to include fine-grained emotions
        with_market_context : bool
            Whether to apply market context
        show_progress : bool
            Whether to show a progress bar
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with advanced sentiment analysis results
        """
        # Convert to list if Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        results = []
        
        # Process in batches
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
                    result = self.analyze_sentiment_advanced(
                        text, model_name=model_name,
                        include_fine_emotions=include_fine_emotions,
                        with_market_context=with_market_context
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing text: {e}")
                    # Add basic result for failed analysis
                    results.append({
                        'original_text': text,
                        'processed_text': text,
                        'sentiment_label': 'neutral',
                        'enhanced_intensity': 'neutral',
                        'error': str(e)
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def analyze_dataframe_advanced(self, df, text_column, result_prefix='sentiment_', 
                                  model_name='best', include_fine_emotions=True,
                                  with_market_context=True):
        """
        Analyze texts in a DataFrame with advanced features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts
        text_column : str
            Column name containing texts
        result_prefix : str
            Prefix for result columns
        model_name : str
            Model to use
        include_fine_emotions : bool
            Whether to include fine-grained emotions
        with_market_context : bool
            Whether to apply market context
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added analysis columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Analyze sentiment for each text
        sentiments = self.batch_analyze_advanced(
            df[text_column], model_name=model_name,
            include_fine_emotions=include_fine_emotions,
            with_market_context=with_market_context
        )
        
        # Add basic columns (same as parent class)
        result_df[f'{result_prefix}label'] = sentiments['sentiment_label']
        result_df[f'{result_prefix}score'] = sentiments['sentiment_score']
        result_df[f'{result_prefix}vader_compound'] = sentiments['vader_compound']
        
        # Add enhanced intensity
        result_df[f'{result_prefix}enhanced_intensity'] = sentiments['enhanced_intensity']
        result_df[f'{result_prefix}intensity_level'] = sentiments['intensity_level']
        
        # Add emotions
        if 'dominant_emotion' in sentiments.columns:
            result_df[f'{result_prefix}dominant_emotion'] = sentiments['dominant_emotion']
            result_df[f'{result_prefix}dominant_emotion_score'] = sentiments['dominant_emotion_score']
        
        # Add fine-grained emotions
        if include_fine_emotions and 'fine_emotions' in sentiments.columns:
            # Extract top 3 fine emotions to separate columns
            result_df[f'{result_prefix}fine_emotions'] = sentiments['fine_emotions'].apply(
                lambda x: ', '.join(list(x.keys())[:3]) if isinstance(x, dict) else ''
            )
            
            # Add columns for specific emotions of interest
            emotion_cols = ['bullish', 'bearish', 'fear', 'joy', 'anger', 'trust']
            for emotion in emotion_cols:
                result_df[f'{result_prefix}emotion_{emotion}'] = sentiments['fine_emotions'].apply(
                    lambda x: x.get(emotion, 0) if isinstance(x, dict) else 0
                )
        
        # Add market interpretation
        if 'market_interpretation' in sentiments.columns:
            result_df[f'{result_prefix}market_interpretation'] = sentiments['market_interpretation']
        
        # Add market context results
        if with_market_context and 'market_context' in sentiments.columns:
            if 'interpretation' in sentiments['market_context'].iloc[0]:
                result_df[f'{result_prefix}market_context'] = sentiments['market_context'].apply(
                    lambda x: x.get('interpretation', '') if isinstance(x, dict) else ''
                )
        
        # Add advanced model results if available
        if 'advanced_sentiment' in sentiments.columns:
            result_df[f'{result_prefix}advanced_sentiment'] = sentiments['advanced_sentiment']
            if 'advanced_confidence' in sentiments.columns:
                result_df[f'{result_prefix}advanced_confidence'] = sentiments['advanced_confidence']
            result_df[f'{result_prefix}advanced_model'] = sentiments['advanced_model']
        
        return result_df
    
    def plot_enhanced_sentiment_distribution(self, results_df, save_path=None, figsize=(12, 8)):
        """
        Plot enhanced sentiment distribution with 7-point scale
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with sentiment analysis results
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Count sentiment labels on enhanced intensity scale
        if 'enhanced_intensity' in results_df.columns:
            sentiment_counts = results_df['enhanced_intensity'].value_counts()
        else:
            sentiment_counts = results_df['sentiment_label'].value_counts()
            
        # Ensure all categories exist and sort them
        intensity_order = list(INTENSITY_COLORS.keys())
        for label in intensity_order:
            if label not in sentiment_counts:
                sentiment_counts[label] = 0
        
        # Sort by intensity
        sentiment_counts = sentiment_counts.reindex(intensity_order)
        
        # Create bar chart with custom gradient colors
        colors = [INTENSITY_COLORS[label] for label in sentiment_counts.index]
        ax = sentiment_counts.plot(kind='bar', color=colors)
        
        # Add count labels on bars
        for i, count in enumerate(sentiment_counts):
            plt.text(i, count + 0.3, str(count), ha='center', va='bottom')
        
        plt.title('Enhanced Sentiment Distribution (7-Point Scale)', fontsize=14)
        plt.xlabel('Sentiment Intensity', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Create a legend with custom labels
        handles = [Patch(facecolor=color, label=label.replace('_', ' ').title()) 
                  for label, color in INTENSITY_COLORS.items()]
        plt.legend(handles=handles, title="Sentiment Intensity", 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_fine_emotion_distribution(self, results_df, save_path=None, figsize=(14, 10)):
        """
        Plot fine-grained emotion distribution
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with sentiment analysis results
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Check which column to use
        if 'fine_emotions' in results_df.columns:
            # Parse the fine_emotions string into a list of emotions
            all_emotions = []
            for emotions_str in results_df['fine_emotions']:
                if isinstance(emotions_str, str) and emotions_str:
                    emotions = [e.strip() for e in emotions_str.split(',')]
                    all_emotions.extend(emotions)
            
            # Count occurrences of each emotion
            emotion_counts = pd.Series(all_emotions).value_counts()
            
            # Filter to keep only emotions with count > 0
            emotion_counts = emotion_counts[emotion_counts > 0]
            
            # Sort by count (descending)
            emotion_counts = emotion_counts.sort_values(ascending=False)
            
            # Limit to top 15 emotions for readability
            if len(emotion_counts) > 15:
                emotion_counts = emotion_counts.iloc[:15]
                
        elif any(col.startswith('emotion_') for col in results_df.columns):
            # Alternative: use the emotion score columns
            emotion_cols = [col for col in results_df.columns if col.startswith('emotion_')]
            emotion_counts = pd.Series({
                col.replace('emotion_', ''): results_df[col].sum()
                for col in emotion_cols
            })
            emotion_counts = emotion_counts[emotion_counts > 0].sort_values(ascending=False)
        else:
            # Fallback: use dominant_emotion if available
            if 'dominant_emotion' in results_df.columns:
                emotion_counts = results_df['dominant_emotion'].value_counts()
            else:
                logger.warning("No emotion data found in DataFrame")
                return
        
        # Create colorful bar chart
        cmap = cm.get_cmap('viridis', len(emotion_counts))
        colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(emotion_counts))]
        
        ax = emotion_counts.plot(kind='bar', color=colors)
        
        # Add count labels on bars
        for i, count in enumerate(emotion_counts):
            plt.text(i, count + 0.1, str(int(count)), ha='center', va='bottom')
        
        plt.title('Fine-Grained Emotion Distribution', fontsize=14)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_emotion_heatmap(self, results_df, save_path=None, figsize=(14, 10)):
        """
        Plot emotion heatmap showing relationships between emotions and sentiment
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with sentiment analysis results
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Get emotion columns
        emotion_cols = [col for col in results_df.columns if col.startswith('emotion_')]
        
        if not emotion_cols:
            logger.warning("No emotion data found for heatmap")
            return
        
        # Prepare data for heatmap: mean emotion score by sentiment category
        if 'enhanced_intensity' in results_df.columns:
            sentiment_col = 'enhanced_intensity'
        else:
            sentiment_col = 'sentiment_label'
        
        # Create pivot table: sentiment categories as rows, emotions as columns
        pivot_data = results_df.pivot_table(
            index=sentiment_col,
            values=emotion_cols,
            aggfunc='mean'
        )
        
        # Rename columns to remove the 'emotion_' prefix
        pivot_data.columns = [col.replace('emotion_', '') for col in pivot_data.columns]
        
        # Create heatmap with custom colormap
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis',
                   linewidths=.5, cbar_kws={'label': 'Mean Emotion Score'})
        
        plt.title('Emotion Intensity by Sentiment Category', fontsize=14)
        plt.ylabel('Sentiment Category', fontsize=12)
        plt.xlabel('Emotion', fontsize=12)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_sentiment_emotion_scatter(self, results_df, x_emotion='fear', y_emotion='optimism', 
                                      save_path=None, figsize=(12, 10)):
        """
        Plot scatter plot of two emotions with sentiment as color
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with sentiment analysis results
        x_emotion : str
            Emotion for x-axis
        y_emotion : str
            Emotion for y-axis
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Get column names for the emotions
        x_col = f'emotion_{x_emotion}' if f'emotion_{x_emotion}' in results_df.columns else None
        y_col = f'emotion_{y_emotion}' if f'emotion_{y_emotion}' in results_df.columns else None
        
        if not x_col or not y_col:
            logger.warning(f"Emotion columns not found: {x_emotion}, {y_emotion}")
            return
        
        # Get sentiment column
        if 'enhanced_intensity' in results_df.columns:
            color_col = 'enhanced_intensity'
        else:
            color_col = 'sentiment_label'
        
        # Create color map for sentiment categories
        if 'enhanced_intensity' in results_df.columns:
            sentiment_colors = INTENSITY_COLORS
            color_map = {cat: sentiment_colors[cat] for cat in results_df[color_col].unique() if cat in sentiment_colors}
        else:
            color_map = {
                'very_positive': '#1E8F4E',
                'positive': '#7ED957',
                'neutral': '#DBDBDB',
                'negative': '#FF6B6B',
                'very_negative': '#D62828'
            }
        
        # Create scatter plot
        for sentiment, color in color_map.items():
            subset = results_df[results_df[color_col] == sentiment]
            plt.scatter(subset[x_col], subset[y_col], c=color, label=sentiment.replace('_', ' ').title(),
                       alpha=0.7, edgecolors='w', linewidths=0.5)
        
        plt.title(f'Emotion Relationship: {x_emotion.title()} vs {y_emotion.title()}', fontsize=14)
        plt.xlabel(f'{x_emotion.title()} Intensity', fontsize=12)
        plt.ylabel(f'{y_emotion.title()} Intensity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add diagonal line
        max_val = max(results_df[x_col].max(), results_df[y_col].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
        
        # Add legend
        plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
    def set_market_context(self, context_dict):
        """
        Set or update market context for contextual analysis
        
        Parameters:
        -----------
        context_dict : dict
            Dictionary with market context information
        """
        self.market_context.update(context_dict)
        logger.info(f"Updated market context: {list(context_dict.keys())}")
    
    def save_enhanced_analyzer(self, filepath):
        """
        Save the enhanced analyzer to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the analyzer
        """
        try:
            # Create a copy without non-picklable objects
            save_obj = EnhancedEmotionalAnalyzer(self.text_processor)
            
            # Copy basic properties
            save_obj.models = self.models
            save_obj.emotion_lexicons = self.emotion_lexicons
            save_obj.extended_emotion_lexicons = self.extended_emotion_lexicons
            save_obj.advanced_models = self.advanced_models
            save_obj.market_context = self.market_context
            
            # Don't save transformer models
            save_obj.emotion_model = None
            save_obj.emotion_tokenizer = None
            save_obj.emotion_pipeline = None
            save_obj.fine_emotion_model = None
            save_obj.fine_emotion_tokenizer = None
            save_obj.fine_emotion_pipeline = None
            save_obj.spacy_nlp = None
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_obj, f)
                
            logger.info(f"Enhanced analyzer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced analyzer: {e}")
    
    @classmethod
    def load_enhanced_analyzer(cls, filepath, load_models=True):
        """
        Load an enhanced analyzer from a file
        
        Parameters:
        -----------
        filepath : str
            Path to load the analyzer from
        load_models : bool
            Whether to load transformer models
            
        Returns:
        --------
        EnhancedEmotionalAnalyzer
            Loaded analyzer
        """
        try:
            with open(filepath, 'rb') as f:
                analyzer = pickle.load(f)
                
            # Initialize VADER
            analyzer.sid = SentimentIntensityAnalyzer()
            
            # Load models if requested
            if load_models:
                analyzer.load_emotion_model()
                try:
                    analyzer.load_fine_emotion_model()
                except Exception as e:
                    logger.warning(f"Could not load fine emotion model: {e}")
                
                if HAS_ADVANCED_NLP:
                    try:
                        import spacy
                        analyzer.spacy_nlp = spacy.load("en_core_web_sm")
                    except Exception as e:
                        logger.warning(f"Could not load spaCy model: {e}")
                
            logger.info(f"Enhanced analyzer loaded from {filepath}")
            return analyzer
            
        except Exception as e:
            logger.error(f"Error loading enhanced analyzer: {e}")
            return cls()  # Return a new instance if loading fails


# Example usage
if __name__ == "__main__":
    # Example financial texts
    example_texts = [
        "I am extremely bullish on $AAPL, their new AI strategy will revolutionize the industry and drive massive growth.",
        "Markets looking very risky, I'm worried about inflation and a potential crash. Considering moving to cash.",
        "Not sure about $AMZN, their growth is slowing but AWS remains strong. Need more data for a clear picture.",
        "I'm outraged by the Fed's latest decision, this will hurt small investors while Wall Street profits!",
        "The latest earnings report exceeded expectations, shows strong guidance and healthy fundamentals.",
        "Tesla's production numbers disappointed again. Bearish on $TSLA until they solve manufacturing issues.",
        "Excited about the new product launch, could be a game-changer for the company's future revenue.",
        "Feeling uncertain about the market's direction with mixed economic signals and geopolitical tensions."
    ]
    
    # Market context example
    market_context = {
        'market_trend': 'bullish',
        'sector': 'Technology',
        'sector_performance': 'outperforming',
        'recent_news_sentiment': 0.62,
        'volatility_index': 'moderate',
        'macro_environment': 'mixed',
        'fed_outlook': 'hawkish'
    }
    
    # Initialize enhanced analyzer
    analyzer = EnhancedEmotionalAnalyzer()
    analyzer.set_market_context(market_context)
    
    # Try to load emotion detection models
    try:
        analyzer.load_emotion_model()
        analyzer.load_fine_emotion_model()
    except Exception as e:
        print(f"Could not load emotion models: {e}")
    
    # Analyze texts
    print("Enhanced Emotional Sentiment Analysis Results:")
    for text in example_texts:
        result = analyzer.analyze_sentiment_advanced(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment_label']} (Score: {result['sentiment_score']:.2f})")
        print(f"Enhanced intensity: {result['enhanced_intensity']} (Level: {result['intensity_level']})")
        
        if 'dominant_emotion' in result:
            print(f"Dominant emotion: {result['dominant_emotion']} ({result['dominant_emotion_score']:.3f})")
        
        if 'fine_emotions' in result and result['fine_emotions']:
            print("Fine-grained emotions:")
            for emotion, score in sorted(result['fine_emotions'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {emotion}: {score:.3f}")
        
        if 'market_interpretation' in result:
            print(f"Market interpretation: {result['market_interpretation']}")
        
        if 'market_context' in result and isinstance(result['market_context'], dict):
            if 'interpretation' in result['market_context']:
                print(f"Contextual analysis: {result['market_context']['interpretation']}")
        
        print("-" * 80)
    
    # Batch analyze and create visualizations
    results = analyzer.batch_analyze_advanced(example_texts)
    
    # Create visualizations
    analyzer.plot_enhanced_sentiment_distribution(results)
    analyzer.plot_fine_emotion_distribution(results)
    analyzer.plot_emotion_heatmap(results)
    analyzer.plot_sentiment_emotion_scatter(results, 'fear', 'optimism')