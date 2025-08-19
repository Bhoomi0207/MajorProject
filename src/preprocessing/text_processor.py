"""
Text preprocessing utilities for stock sentiment analysis
"""
import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

class StockTextProcessor:
    """
    Class for preprocessing and analyzing text data related to stocks
    """
    def __init__(self, use_stemming=False):
        """
        Initialize text processor
        
        Parameters:
        -----------
        use_stemming : bool
            Whether to use stemming (True) or lemmatization (False)
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_stemming = use_stemming
        self.sid = SentimentIntensityAnalyzer()
        
        # Add finance-specific stop words
        finance_stop_words = {
            'stock', 'stocks', 'market', 'markets', 'share', 'shares',
            'price', 'prices', 'trading', 'trade', 'trades', 'trader', 'traders',
            'company', 'companies', 'corporation', 'inc', 'corp', 'llc',
            'financial', 'finance', 'investment', 'investments', 'investor', 'investors',
            'reuters', 'bloomberg', 'wsj', 'cnbc', 'news'
        }
        self.stop_words.update(finance_stop_words)
        
        # Stock-specific sentiment modifiers
        self.stock_financial_lexicon = {
            # Bullish terms with sentiment scores
            'buy': 0.6, 'strong buy': 0.8, 'outperform': 0.7, 'overweight': 0.6, 
            'upgrade': 0.7, 'upgraded': 0.7, 'bullish': 0.8, 'upside': 0.6,
            'beat': 0.5, 'beats': 0.5, 'exceeded': 0.5, 'exceeds': 0.5,
            'growth': 0.4, 'growing': 0.4, 'grew': 0.4, 'expand': 0.4,
            'surge': 0.6, 'surged': 0.6, 'rally': 0.5, 'rallied': 0.5,
            'gain': 0.4, 'gains': 0.4, 'climb': 0.4, 'climbs': 0.4,
            'positive': 0.5, 'stronger': 0.5, 'strength': 0.4, 'opportunity': 0.4,
            'breakout': 0.6, 'momentum': 0.4, 'uptrend': 0.6, 'winner': 0.5,
            
            # Bearish terms with sentiment scores
            'sell': -0.6, 'strong sell': -0.8, 'underperform': -0.7, 'underweight': -0.6,
            'downgrade': -0.7, 'downgraded': -0.7, 'bearish': -0.8, 'downside': -0.6,
            'miss': -0.5, 'misses': -0.5, 'missed': -0.5, 'disappointing': -0.6,
            'decline': -0.5, 'declining': -0.5, 'decreased': -0.5, 'shrink': -0.5,
            'drop': -0.5, 'drops': -0.5, 'plunge': -0.7, 'plunged': -0.7, 
            'fall': -0.5, 'falls': -0.5, 'fell': -0.5, 'sink': -0.6, 'sinks': -0.6,
            'negative': -0.5, 'weaker': -0.5, 'weakness': -0.4, 'risk': -0.4,
            'breakdown': -0.6, 'downturn': -0.5, 'downtrend': -0.6, 'loser': -0.5
        }
    
    def clean_text(self, text, remove_urls=True, remove_numbers=True, 
                 remove_special_chars=True, expand_contractions=True):
        """
        Basic text cleaning
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters
        if remove_special_chars:
            text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Expand contractions (basic)
        if expand_contractions:
            contractions = {
                "n't": " not", "'s": " is", "'m": " am", "'re": " are",
                "'ll": " will", "'ve": " have", "'d": " would"
            }
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_preprocess(self, text, remove_stopwords=True, 
                              normalize_words=True, min_word_length=2):
        """
        Tokenize and preprocess text
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(cleaned_text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Filter by minimum length
        tokens = [token for token in tokens if len(token) >= min_word_length]
        
        # Normalize words (stem or lemmatize)
        if normalize_words:
            if self.use_stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]
            else:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def extract_stock_mentions(self, text, stock_tickers=None):
        """
        Extract mentions of stock tickers from text
        
        Parameters:
        -----------
        text : str
            Text to analyze
        stock_tickers : list
            List of stock tickers to look for (if None, will use common patterns)
            
        Returns:
        --------
        list
            List of found stock tickers
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # If no specific tickers provided, use a pattern to find potential tickers
        if stock_tickers is None:
            # Look for $TICKER format
            dollar_pattern = r'\$([A-Za-z]{1,5})'
            dollar_matches = re.findall(dollar_pattern, text)
            
            # Look for common ticker formats (1-5 capital letters)
            ticker_pattern = r'\b([A-Z]{1,5})\b'
            ticker_matches = re.findall(ticker_pattern, text)
            
            return list(set(dollar_matches + ticker_matches))
        else:
            # Look for specific tickers
            mentioned_tickers = []
            for ticker in stock_tickers:
                # Match ticker with $ prefix
                if f"${ticker.lower()}" in text.lower() or f"${ticker}" in text:
                    mentioned_tickers.append(ticker)
                # Match ticker as standalone word
                elif re.search(rf'\b{ticker}\b', text, re.IGNORECASE):
                    mentioned_tickers.append(ticker)
            
            return mentioned_tickers
    
    def analyze_sentiment(self, text, stock_context=None):
        """
        Analyze sentiment of text with stock-specific adjustments
        
        Parameters:
        -----------
        text : str
            Text to analyze
        stock_context : str or None
            Stock ticker for context-specific adjustments
            
        Returns:
        --------
        dict
            Dictionary of sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 1.0,
                'stock_specific_score': 0.0
            }
        
        # Get base sentiment from VADER
        base_sentiment = self.sid.polarity_scores(text)
        
        # Apply stock-specific lexicon adjustments
        stock_specific_score = 0.0
        word_count = 0
        
        # Tokenize text for lexicon matching
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        
        # Create bigrams for multi-word matching
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens) - 1)]
        
        # Check for lexicon terms
        for term in tokens + bigrams:
            if term in self.stock_financial_lexicon:
                stock_specific_score += self.stock_financial_lexicon[term]
                word_count += 1
        
        # Normalize stock-specific score
        if word_count > 0:
            stock_specific_score = stock_specific_score / word_count
        
        # If a specific stock ticker is provided, check for positive/negative context
        if stock_context:
            if stock_context.lower() in cleaned_text.lower():
                # If ticker is mentioned with positive sentiment, boost the score
                if base_sentiment['compound'] > 0.2:
                    stock_specific_score += 0.1
                # If ticker is mentioned with negative sentiment, lower the score
                elif base_sentiment['compound'] < -0.2:
                    stock_specific_score -= 0.1
        
        # Add stock-specific score to results
        result = base_sentiment.copy()
        result['stock_specific_score'] = stock_specific_score
        
        # Calculate adjusted compound score (blend of VADER and stock-specific)
        result['adjusted_compound'] = 0.7 * base_sentiment['compound'] + 0.3 * stock_specific_score
        
        return result
    
    def analyze_text_batch(self, texts, stock_context=None):
        """
        Analyze a batch of texts
        
        Parameters:
        -----------
        texts : list
            List of texts to analyze
        stock_context : str or None
            Stock ticker for context-specific adjustments
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with analysis results
        """
        results = []
        
        for text in texts:
            # Clean text
            cleaned_text = self.clean_text(text)
            
            # Get tokens
            tokens = self.tokenize_and_preprocess(cleaned_text)
            
            # Extract stock mentions
            stock_mentions = self.extract_stock_mentions(text)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(cleaned_text, stock_context)
            
            # Create result entry
            result = {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'token_count': len(tokens),
                'stock_mentions': ','.join(stock_mentions),
                'sentiment_compound': sentiment['compound'],
                'sentiment_positive': sentiment['pos'],
                'sentiment_negative': sentiment['neg'],
                'sentiment_neutral': sentiment['neu'],
                'stock_specific_score': sentiment['stock_specific_score'],
                'adjusted_compound': sentiment['adjusted_compound']
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def process_dataframe_text(self, df, text_column, stock_context=None, output_prefix=''):
        """
        Process text in a DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing text data
        text_column : str
            Name of column containing text to process
        stock_context : str or None
        stock_context : str or None
            Stock ticker for context-specific adjustments
        output_prefix : str
            Prefix for output column names
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added text analysis columns
        """
        if not isinstance(df, pd.DataFrame) or text_column not in df.columns:
            logger.error(f"Invalid DataFrame or text column '{text_column}' not found")
            return df
        
        logger.info(f"Processing text in column '{text_column}' for {len(df)} rows")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add column with cleaned text
        clean_col = f"{output_prefix}cleaned_text"
        result_df[clean_col] = result_df[text_column].apply(lambda x: self.clean_text(x) if isinstance(x, str) else "")
        
        # Add token count column
        token_col = f"{output_prefix}token_count"
        result_df[token_col] = result_df[clean_col].apply(
            lambda x: len(self.tokenize_and_preprocess(x)) if isinstance(x, str) else 0
        )
        
        # Extract stock mentions
        mentions_col = f"{output_prefix}stock_mentions"
        result_df[mentions_col] = result_df[text_column].apply(
            lambda x: ','.join(self.extract_stock_mentions(x)) if isinstance(x, str) else ""
        )
        
        # Analyze sentiment
        sentiment_results = result_df[clean_col].apply(
            lambda x: self.analyze_sentiment(x, stock_context) if isinstance(x, str) else {
                'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 
                'stock_specific_score': 0.0, 'adjusted_compound': 0.0
            }
        )
        
        # Add sentiment columns
        result_df[f"{output_prefix}sentiment_compound"] = sentiment_results.apply(lambda x: x['compound'])
        result_df[f"{output_prefix}sentiment_positive"] = sentiment_results.apply(lambda x: x['pos'])
        result_df[f"{output_prefix}sentiment_negative"] = sentiment_results.apply(lambda x: x['neg'])
        result_df[f"{output_prefix}sentiment_neutral"] = sentiment_results.apply(lambda x: x['neu'])
        result_df[f"{output_prefix}stock_specific_score"] = sentiment_results.apply(lambda x: x['stock_specific_score'])
        result_df[f"{output_prefix}adjusted_compound"] = sentiment_results.apply(lambda x: x['adjusted_compound'])
        
        return result_df