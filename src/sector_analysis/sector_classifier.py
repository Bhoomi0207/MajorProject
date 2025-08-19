import numpy as np
import pandas as pd
import logging
import re
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import joblib
from collections import Counter

from src.preprocessing.text_processor import StockTextProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectorClassifier:
    """
    Class for classifying financial texts into market sectors or industries
    """
    
    def __init__(self, text_processor=None):
        """
        Initialize the sector classifier
        
        Parameters:
        -----------
        text_processor : StockTextProcessor, optional
            Text processor to use for preprocessing. Creates a new one if None
        """
        self.text_processor = text_processor if text_processor is not None else StockTextProcessor()
        self.models = {}
        self.vectorizer = None
        self.label_encoder = None
        
        # Common market sectors
        self.default_sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
            'Consumer Defensive', 'Industrials', 'Communication Services',
            'Energy', 'Basic Materials', 'Real Estate', 'Utilities'
        ]
        
        # Keywords associated with each sector for rule-based classification
        self.sector_keywords = {
            'Technology': [
                'tech', 'software', 'hardware', 'semiconductor', 'cloud', 'ai', 'artificial intelligence',
                'machine learning', 'data', 'internet', 'computer', 'cybersecurity', 'saas', 'iot',
                'blockchain', 'fintech', 'digital', 'ecommerce', 'automation'
            ],
            'Healthcare': [
                'health', 'pharma', 'pharmaceutical', 'biotech', 'medical', 'drug', 'therapy',
                'therapeutic', 'clinical', 'hospital', 'healthcare', 'medicine', 'patient',
                'vaccine', 'diagnostic', 'treatment'
            ],
            'Financial Services': [
                'bank', 'financial', 'insurance', 'credit', 'payment', 'loan', 'mortgage',
                'invest', 'capital', 'asset', 'wealth', 'stock', 'bond', 'equity', 'fund',
                'broker', 'exchange', 'finance'
            ],
            'Consumer Cyclical': [
                'retail', 'apparel', 'restaurant', 'automobile', 'hotel', 'travel', 'leisure',
                'entertainment', 'luxury', 'home', 'furniture', 'consumer discretionary',
                'e-commerce', 'fashion'
            ],
            'Consumer Defensive': [
                'food', 'beverage', 'grocery', 'household', 'personal care', 'consumer staple',
                'tobacco', 'discount', 'supermarket'
            ],
            'Industrials': [
                'industrial', 'manufacturing', 'aerospace', 'defense', 'construction', 'machinery',
                'transportation', 'logistics', 'airline', 'rail', 'shipping', 'waste', 'infrastructure'
            ],
            'Communication Services': [
                'telecom', 'telecommunications', 'media', 'advertising', 'publishing', 'broadcasting',
                'social media', 'streaming', 'entertainment', 'communication'
            ],
            'Energy': [
                'oil', 'gas', 'petroleum', 'renewable', 'solar', 'wind', 'energy', 'power',
                'utility', 'electricity', 'nuclear', 'coal', 'refining', 'drilling'
            ],
            'Basic Materials': [
                'mining', 'metal', 'steel', 'chemical', 'agriculture', 'forestry', 'paper',
                'packaging', 'commodity', 'gold', 'silver', 'copper', 'uranium'
            ],
            'Real Estate': [
                'property', 'reit', 'real estate', 'building', 'commercial property',
                'residential', 'apartment', 'housing', 'mortgage'
            ],
            'Utilities': [
                'utility', 'electric', 'water', 'gas utility', 'power generation',
                'energy distribution', 'waste management'
            ]
        }
    
    def classify_by_keywords(self, text):
        """
        Classify text into sectors using keyword matching
        
        Parameters:
        -----------
        text : str
            Text to classify
            
        Returns:
        --------
        list
            List of likely sectors based on keyword matching, sorted by match count
        """
        # Preprocess the text
        processed_text = self.text_processor.preprocess_text(text)
        
        # Count keyword matches for each sector
        sector_matches = {}
        
        for sector, keywords in self.sector_keywords.items():
            match_count = 0
            for keyword in keywords:
                # Look for the keyword as a whole word
                matches = re.findall(r'\b{}\b'.format(re.escape(keyword)), processed_text.lower())
                match_count += len(matches)
            
            if match_count > 0:
                sector_matches[sector] = match_count
        
        # Sort sectors by match count (descending)
        sorted_sectors = sorted(sector_matches.items(), key=lambda x: x[1], reverse=True)
        
        # Extract just the sector names
        sectors = [sector for sector, count in sorted_sectors]
        
        # If no matches, return 'Unknown'
        if not sectors:
            return ['Unknown']
        
        return sectors
    
    def train_classifier(self, texts, sectors, model_type='logistic_regression', test_size=0.2):
        """
        Train a sector classifier
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts for training
        sectors : list or pandas.Series
            Sector labels
        model_type : str
            Type of model to train ('logistic_regression', 'random_forest', 'naive_bayes', 'svm')
        test_size : float
            Fraction of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary with model, metrics, and configuration
        """
        # Convert to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(sectors, pd.Series):
            sectors = sectors.tolist()
        
        # Preprocess texts
        processed_texts = [self.text_processor.preprocess_text(text) for text in texts]
        
        # Encode sector labels
        self.label_encoder = LabelEncoder()
        encoded_sectors = self.label_encoder.fit_transform(sectors)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, encoded_sectors, test_size=test_size, random_state=42, stratify=encoded_sectors
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        # Create model
        if model_type == 'logistic_regression':
            model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'naive_bayes':
            model = MultinomialNB(alpha=1.0)
        elif model_type == 'svm':
            model = LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=10000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('model', model)
        ])
        
        # Train model
        logger.info(f"Training {model_type} sector classifier")
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Create and store model information
        model_info = {
            'pipeline': pipeline,
            'model_type': model_type,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'classes': self.label_encoder.classes_,
            'label_encoder': self.label_encoder,
            'vectorizer': self.vectorizer
        }
        
        # Store model in instance
        self.models[model_type] = model_info
        
        return model_info
    
    def optimize_hyperparameters(self, texts, sectors):
        """
        Find optimal hyperparameters for the sector classifier
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts for training
        sectors : list or pandas.Series
            Sector labels
            
        Returns:
        --------
        dict
            Dictionary with optimized model and hyperparameters
        """
        # Convert to list if needed
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if isinstance(sectors, pd.Series):
            sectors = sectors.tolist()
        
        # Preprocess texts
        processed_texts = [self.text_processor.preprocess_text(text) for text in texts]
        
        # Encode sector labels
        self.label_encoder = LabelEncoder()
        encoded_sectors = self.label_encoder.fit_transform(sectors)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, encoded_sectors, test_size=0.2, random_state=42, stratify=encoded_sectors
        )
        
        # Create pipeline with TF-IDF and LogisticRegression
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('model', LogisticRegression(class_weight='balanced', random_state=42))
        ])
        
        # Define parameter grid
        param_grid = {
            'vectorizer__max_features': [5000, 10000, 15000],
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'model__C': [0.1, 1.0, 10.0],
            'model__solver': ['liblinear', 'saga']
        }
        
        # Create grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )
        
        # Perform grid search
        logger.info("Starting hyperparameter optimization")
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        
        logger.info(f"Best accuracy: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Evaluate best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Best model test accuracy: {accuracy:.4f}")
        
        # Create and store model information
        model_info = {
            'pipeline': best_model,
            'model_type': 'optimized_logistic_regression',
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'classes': self.label_encoder.classes_,
            'best_params': best_params,
            'label_encoder': self.label_encoder,
            'vectorizer': best_model.named_steps['vectorizer']
        }
        
        # Store model in instance
        self.models['optimized'] = model_info
        
        return model_info
    
    def classify_sector(self, text, model_key=None):
        """
        Classify text into a market sector
        
        Parameters:
        -----------
        text : str
            Text to classify
        model_key : str, optional
            Key of model to use. If None, uses first available model or keyword matching
            
        Returns:
        --------
        dict
            Classification results with sector and confidence
        """
        # If no model key provided, use first available model
        if model_key is None and self.models:
            model_key = list(self.models.keys())[0]
        
        # If no models available, use keyword matching
        if model_key is None or model_key not in self.models:
            logger.info("No trained model available, using keyword matching")
            sectors = self.classify_by_keywords(text)
            return {
                'sector': sectors[0] if sectors else 'Unknown',
                'all_sectors': sectors,
                'method': 'keyword',
                'confidence': None
            }
        
        # Get model info
        model_info = self.models[model_key]
        pipeline = model_info['pipeline']
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(text)
        
        try:
            # Predict sector
            prediction = pipeline.predict([processed_text])[0]
            sector = model_info['classes'][prediction]
            
            # Get prediction probability if available
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba([processed_text])[0]
                confidence = probabilities[prediction]
            else:
                # For models like LinearSVC that don't have predict_proba
                confidence = None
                
            # Get keyword classification as fallback/validation
            keyword_sectors = self.classify_by_keywords(text)
            
            return {
                'sector': sector,
                'confidence': confidence,
                'keyword_sectors': keyword_sectors,
                'method': 'model',
                'model_used': model_key
            }
        except Exception as e:
            logger.error(f"Error predicting sector: {e}")
            sectors = self.classify_by_keywords(text)
            return {
                'sector': sectors[0] if sectors else 'Unknown',
                'all_sectors': sectors,
                'method': 'keyword',
                'confidence': None,
                'error': str(e)
            }
    
    def batch_classify(self, texts, model_key=None):
        """
        Classify sectors for a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to classify
        model_key : str, optional
            Key of model to use
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector classifications for each text
        """
        results = []
        for text in texts:
            result = self.classify_sector(text, model_key)
            results.append(result)
        return pd.DataFrame(results)
    
    def analyze_dataframe(self, df, text_column, model_key=None, result_prefix='sector_'):
        """
        Classify sectors for texts in a DataFrame column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to classify
        text_column : str
            Name of column containing texts
        model_key : str, optional
            Key of model to use
        result_prefix : str
            Prefix for new columns with classification results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sector classification columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Classify sectors for each text
        classifications = self.batch_classify(df[text_column], model_key)
        
        # Add classification columns to result DataFrame
        result_df[f'{result_prefix}classification'] = classifications['sector']
        
        if 'confidence' in classifications.columns:
            # Only add confidence if available
            confidences = classifications['confidence'].copy()
            result_df[f'{result_prefix}confidence'] = confidences
        
        if 'keyword_sectors' in classifications.columns:
            result_df[f'{result_prefix}keywords'] = classifications['keyword_sectors']
            
        if 'method' in classifications.columns:
            result_df[f'{result_prefix}method'] = classifications['method']
        
        return result_df
    
    def plot_sector_distribution(self, sectors):
        """
        Plot the distribution of sectors
        
        Parameters:
        -----------
        sectors : list or pandas.Series
            Sector classifications
        """
        # Count sectors
        if isinstance(sectors, pd.Series):
            sector_counts = sectors.value_counts()
        else:
            sector_counts = pd.Series(Counter(sectors))
        
        # Sort by count (descending)
        sector_counts = sector_counts.sort_values(ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = sns.barplot(x=sector_counts.index, y=sector_counts.values)
        
        # Add count labels on top of bars
        for i, count in enumerate(sector_counts.values):
            bars.text(i, count + 0.5, str(count), ha='center')
        
        plt.title('Distribution of Market Sectors')
        plt.xlabel('Sector')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('analysis', 'sector_distribution.png')
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
        
        # Normalize confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=model_info['classes'],
                   yticklabels=model_info['classes'])
        plt.title(f'Normalized Confusion Matrix for {model_key}')
        plt.ylabel('True Sector')
        plt.xlabel('Predicted Sector')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('models', f'{model_key}_confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def get_top_keywords(self, sector, n=20):
        """
        Get top keywords associated with a sector from the trained model
        
        Parameters:
        -----------
        sector : str
            Sector to get keywords for
        n : int
            Number of top keywords to return
            
        Returns:
        --------
        list
            List of (keyword, weight) tuples
        """
        # Check if we have any trained models
        if not self.models:
            logger.warning("No trained models available")
            return self.sector_keywords.get(sector, [])
        
        # Use the first available model
        model_key = list(self.models.keys())[0]
        model_info = self.models[model_key]
        
        # Check if model has feature names
        if 'vectorizer' not in model_info:
            logger.warning("Model doesn't have vectorizer information")
            return self.sector_keywords.get(sector, [])
        
        vectorizer = model_info['vectorizer']
        feature_names = vectorizer.get_feature_names_out()
        
        # Get label encoder and find sector index
        label_encoder = model_info['label_encoder']
        try:
            sector_idx = np.where(label_encoder.classes_ == sector)[0][0]
        except:
            logger.warning(f"Sector {sector} not found in trained model")
            return self.sector_keywords.get(sector, [])
        
        # Check if model coefficients are available (for linear models)
        model = model_info['pipeline'].named_steps['model']
        
        try:
            if hasattr(model, 'coef_'):
                # For linear models like LogisticRegression, LinearSVC
                coefficients = model.coef_[sector_idx]
            elif hasattr(model, 'feature_importances_'):
                # For tree-based models like RandomForest
                coefficients = model.feature_importances_
            else:
                logger.warning("Model doesn't have accessible feature weights")
                return self.sector_keywords.get(sector, [])
            
            # Get top keywords by coefficient/importance
            top_indices = np.argsort(coefficients)[-n:][::-1]
            top_keywords = [(feature_names[i], coefficients[i]) for i in top_indices]
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error getting top keywords: {e}")
            return self.sector_keywords.get(sector, [])
    
    def visualize_sector_keywords(self, sector):
        """
        Visualize top keywords for a sector
        
        Parameters:
        -----------
        sector : str
            Sector to visualize keywords for
        """
        # Get top keywords
        top_keywords = self.get_top_keywords(sector, n=20)
        
        if not top_keywords:
            logger.warning(f"No keywords found for sector {sector}")
            return
        
        # Extract keywords and weights
        keywords = [kw[0] for kw in top_keywords]
        weights = [kw[1] for kw in top_keywords]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 8))
        bars = plt.barh(keywords, weights, color='skyblue')
        
        # Add weight labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{weights[i]:.4f}", va='center')
        
        plt.title(f'Top Keywords for {sector}')
        plt.xlabel('Importance Weight')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('analysis', f'keywords_{sector.lower().replace(" ", "_")}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()
    
    def get_sector_tickers(self, sector):
        """
        Get commonly referenced ticker symbols for a sector
        
        Parameters:
        -----------
        sector : str
            Sector to get tickers for
            
        Returns:
        --------
        list
            List of (ticker, stock_name) tuples
        """
        # Pre-defined mapping of sectors to major stocks
        sector_tickers = {
            'Technology': [
                ('AAPL', 'Apple Inc.'), ('MSFT', 'Microsoft Corp.'), ('GOOGL', 'Alphabet Inc.'),
                ('AMZN', 'Amazon.com Inc.'), ('TSLA', 'Tesla Inc.'), ('NVDA', 'NVIDIA Corp.'),
                ('AMD', 'Advanced Micro Devices'), ('INTC', 'Intel Corp.'), ('CRM', 'Salesforce'),
                ('CSCO', 'Cisco Systems')
            ],
            'Healthcare': [
                ('JNJ', 'Johnson & Johnson'), ('PFE', 'Pfizer Inc.'), ('UNH', 'UnitedHealth Group'),
                ('MRK', 'Merck & Co.'), ('ABBV', 'AbbVie Inc.'), ('LLY', 'Eli Lilly & Co.'),
                ('BMY', 'Bristol-Myers Squibb'), ('AMGN', 'Amgen Inc.'), ('GILD', 'Gilead Sciences'),
                ('CVS', 'CVS Health Corp.')
            ],
            'Financial Services': [
                ('JPM', 'JPMorgan Chase & Co.'), ('BAC', 'Bank of America Corp.'),
                ('WFC', 'Wells Fargo & Co.'), ('C', 'Citigroup Inc.'), ('GS', 'Goldman Sachs Group'),
                ('MS', 'Morgan Stanley'), ('V', 'Visa Inc.'), ('MA', 'Mastercard Inc.'),
                ('AXP', 'American Express'), ('BLK', 'BlackRock Inc.')
            ],
            'Consumer Cyclical': [
                ('HD', 'Home Depot Inc.'), ('NKE', 'Nike Inc.'), ('MCD', 'McDonald\'s Corp.'),
                ('SBUX', 'Starbucks Corp.'), ('DIS', 'Walt Disney Co.'), ('TGT', 'Target Corp.'),
                ('LOW', 'Lowe\'s Companies'), ('BKNG', 'Booking Holdings'), ('F', 'Ford Motor Co.'),
                ('GM', 'General Motors Co.')
            ],
            'Consumer Defensive': [
                ('WMT', 'Walmart Inc.'), ('PG', 'Procter & Gamble'), ('KO', 'Coca-Cola Co.'),
                ('PEP', 'PepsiCo Inc.'), ('COST', 'Costco Wholesale'), ('PM', 'Philip Morris'),
                ('MO', 'Altria Group Inc.'), ('EL', 'Estée Lauder'), ('CL', 'Colgate-Palmolive'),
                ('GIS', 'General Mills Inc.')
            ],
            'Industrials': [
                ('BA', 'Boeing Co.'), ('HON', 'Honeywell International'), ('UPS', 'United Parcel Service'),
                ('UNP', 'Union Pacific Corp.'), ('RTX', 'Raytheon Technologies'), ('CAT', 'Caterpillar Inc.'),
                ('DE', 'Deere & Co.'), ('MMM', '3M Co.'), ('GE', 'General Electric'), 
                ('LMT', 'Lockheed Martin')
            ],
            'Communication Services': [
                ('FB', 'Meta Platforms'), ('NFLX', 'Netflix Inc.'), ('CMCSA', 'Comcast Corp.'),
                ('T', 'AT&T Inc.'), ('VZ', 'Verizon Communications'), ('TMUS', 'T-Mobile US'),
                ('DIS', 'Walt Disney Co.'), ('TWTR', 'Twitter Inc.'), ('EA', 'Electronic Arts'),
                ('ATVI', 'Activision Blizzard')
            ],
            'Energy': [
                ('XOM', 'Exxon Mobil Corp.'), ('CVX', 'Chevron Corp.'), ('COP', 'ConocoPhillips'),
                ('SLB', 'Schlumberger NV'), ('EOG', 'EOG Resources'), ('MPC', 'Marathon Petroleum'),
                ('PSX', 'Phillips 66'), ('OXY', 'Occidental Petroleum'), ('VLO', 'Valero Energy'),
                ('PXD', 'Pioneer Natural Resources')
            ],
            'Basic Materials': [
                ('LIN', 'Linde plc'), ('APD', 'Air Products & Chemicals'), ('DOW', 'Dow Inc.'),
                ('DD', 'DuPont de Nemours'), ('FCX', 'Freeport-McMoRan'), ('NEM', 'Newmont Corp.'),
                ('NUE', 'Nucor Corp.'), ('ECL', 'Ecolab Inc.'), ('GOLD', 'Barrick Gold Corp.'),
                ('SHW', 'Sherwin-Williams')
            ],
            'Real Estate': [
                ('AMT', 'American Tower'), ('PLD', 'Prologis Inc.'), ('CCI', 'Crown Castle'),
                ('SPG', 'Simon Property Group'), ('EQIX', 'Equinix Inc.'), ('PSA', 'Public Storage'),
                ('O', 'Realty Income Corp.'), ('WELL', 'Welltower Inc.'), ('AVB', 'AvalonBay Communities'),
                ('EQR', 'Equity Residential')
            ],
            'Utilities': [
                ('NEE', 'NextEra Energy'), ('DUK', 'Duke Energy Corp.'), ('SO', 'Southern Company'),
                ('D', 'Dominion Energy'), ('AEP', 'American Electric Power'), ('EXC', 'Exelon Corp.'),
                ('PCG', 'PG&E Corp.'), ('SRE', 'Sempra Energy'), ('XEL', 'Xcel Energy Inc.'),
                ('WEC', 'WEC Energy Group')
            ]
        }
        
        return sector_tickers.get(sector, [])
    
    def save_classifier(self, filepath):
        """
        Save the classifier to a file
        
        Parameters:
        -----------
        filepath : str
            File path to save the classifier to
        """
        try:
            # Save using joblib for better handling of sklearn objects
            joblib.dump(self, filepath)
            logger.info(f"Sector classifier saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving classifier: {e}")
    
    @classmethod
    def load_classifier(cls, filepath):
        """
        Load a saved classifier from file
        
        Parameters:
        -----------
        filepath : str
            File path to load the classifier from
            
        Returns:
        --------
        SectorClassifier
            Loaded classifier
        """
        try:
            classifier = joblib.load(filepath)
            logger.info(f"Sector classifier loaded from {filepath}")
            return classifier
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            return cls()  # Return a new instance if loading fails


class IndustryClassifier(SectorClassifier):
    """
    Class for classifying financial texts into industries within market sectors
    Provides more granular classification than SectorClassifier
    """
    
    def __init__(self, text_processor=None):
        """
        Initialize the industry classifier
        
        Parameters:
        -----------
        text_processor : StockTextProcessor, optional
            Text processor to use for preprocessing. Creates a new one if None
        """
        super().__init__(text_processor)
        
        # Industry groups within sectors
        self.industry_groups = {
            'Technology': [
                'Software', 'Hardware', 'Semiconductors', 'IT Services', 
                'Electronic Components', 'Consumer Electronics', 'Cloud Services',
                'Cybersecurity', 'AI & Machine Learning', 'Fintech'
            ],
            'Healthcare': [
                'Pharmaceuticals', 'Biotechnology', 'Medical Devices', 'Healthcare Services',
                'Health Insurance', 'Hospitals', 'Diagnostics', 'Life Sciences Tools',
                'Healthcare Technology', 'Managed Care'
            ],
            'Financial Services': [
                'Banks', 'Investment Banking', 'Asset Management', 'Insurance',
                'Consumer Finance', 'Financial Exchanges', 'Mortgage Finance', 
                'Financial Data Services', 'Diversified Financial Services', 'REITs'
            ],
            'Consumer Cyclical': [
                'Retail', 'Automotive', 'Hotels & Entertainment', 'Apparel & Luxury',
                'Restaurants', 'Home Improvement', 'Travel & Leisure', 'Media',
                'E-Commerce', 'Housing & Construction'
            ],
            'Consumer Defensive': [
                'Food Products', 'Beverages', 'Household Products', 'Personal Care',
                'Tobacco', 'Discount Stores', 'Food Distribution', 'Packaged Foods',
                'Agricultural Products', 'Drug Stores'
            ],
            'Industrials': [
                'Aerospace & Defense', 'Construction & Engineering', 'Machinery',
                'Transportation & Logistics', 'Business Services', 'Electrical Equipment',
                'Commercial Services', 'Industrial Distribution', 'Airlines',
                'Waste Management'
            ],
            'Communication Services': [
                'Telecommunication', 'Media & Broadcasting', 'Social Media',
                'Advertising', 'Publishing', 'Entertainment Content', 'Video Gaming',
                'Internet Services', 'Cable & Satellite', 'Wireless Services'
            ],
            'Energy': [
                'Oil & Gas E&P', 'Oil & Gas Equipment', 'Oil & Gas Midstream',
                'Oil & Gas Refining', 'Oil & Gas Integrated', 'Renewable Energy',
                'Alternative Energy', 'Coal', 'Energy Services', 'Utilities'
            ],
            'Basic Materials': [
                'Chemicals', 'Metals & Mining', 'Steel', 'Paper & Forest Products',
                'Agricultural Inputs', 'Specialty Chemicals', 'Gold', 'Building Materials',
                'Copper', 'Aluminum'
            ],
            'Real Estate': [
                'Residential', 'Commercial', 'Industrial REITs', 'Retail REITs',
                'Office REITs', 'Healthcare REITs', 'Hotel & Resort REITs',
                'Real Estate Services', 'Real Estate Development', 'Specialty REITs'
            ],
            'Utilities': [
                'Electric Utilities', 'Gas Utilities', 'Water Utilities',
                'Multi-Utilities', 'Independent Power Producers', 'Renewable Utilities',
                'Regulated Electric', 'Regulated Gas', 'Regulated Water'
            ]
        }
        
        # Industry-specific keywords for classification
        self.industry_keywords = {
            # Technology industries
            'Software': [
                'software', 'application', 'SaaS', 'platform', 'operating system', 'code',
                'programming', 'developer', 'enterprise software', 'app'
            ],
            'Semiconductors': [
                'semiconductor', 'chip', 'processor', 'integrated circuit', 'wafer', 'foundry',
                'memory', 'gpu', 'microcontroller', 'fab', 'node'
            ],
            'Cloud Services': [
                'cloud', 'aws', 'azure', 'iaas', 'paas', 'saas', 'data center',
                'hosting', 'serverless', 'cloud computing', 'cloud storage'
            ],
            
            # Healthcare industries
            'Pharmaceuticals': [
                'pharmaceutical', 'drug', 'medicine', 'prescription', 'compound',
                'formulation', 'patent', 'pipeline', 'fda approval', 'generic'
            ],
            'Biotechnology': [
                'biotech', 'gene', 'dna', 'therapy', 'cell therapy', 'genomics',
                'biologic', 'clinical trial', 'antibody', 'immunotherapy'
            ],
            'Medical Devices': [
                'device', 'implant', 'equipment', 'diagnostic device', 'monitoring',
                'wearable', 'imaging', 'surgical device', 'medical instrument'
            ],
            
            # Financial Services industries
            'Banks': [
                'bank', 'lending', 'loan', 'deposit', 'credit', 'branch', 'retail banking',
                'commercial banking', 'investment banking', 'interest rate'
            ],
            'Insurance': [
                'insurance', 'premium', 'policy', 'underwriting', 'claim', 'risk',
                'insurer', 'reinsurance', 'actuary', 'coverage'
            ],
            'Asset Management': [
                'asset management', 'fund', 'portfolio', 'investment', 'wealth',
                'aum', 'etf', 'mutual fund', 'institutional investor', 'hedge fund'
            ],
            
            # Rest of industry keywords can be expanded as needed for each industry
        }
        
    def classify_industry(self, text, model_key=None):
        """
        Classify text into an industry group
        
        Parameters:
        -----------
        text : str
            Text to classify
        model_key : str, optional
            Key of model to use
            
        Returns:
        --------
        dict
            Classification results with sector, industry, and confidence
        """
        # First get the sector classification
        sector_result = self.classify_sector(text, model_key)
        sector = sector_result['sector']
        
        # Get possible industries for this sector
        possible_industries = self.industry_groups.get(sector, [])
        
        if not possible_industries:
            # If sector has no defined industries or sector is Unknown
            return {
                'sector': sector,
                'industry': 'Unknown',
                'method': sector_result['method'],
                'confidence': sector_result.get('confidence')
            }
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(text)
        
        # Count keyword matches for each industry in this sector
        industry_matches = {}
        
        for industry in possible_industries:
            match_count = 0
            
            # Use industry keywords if available
            keywords = self.industry_keywords.get(industry, [])
            
            if not keywords:
                # If no specific keywords for this industry, use the industry name itself
                keywords = [industry.lower()]
            
            for keyword in keywords:
                # Look for the keyword as a whole word
                matches = re.findall(r'\b{}\b'.format(re.escape(keyword)), processed_text.lower())
                match_count += len(matches)
            
            if match_count > 0:
                industry_matches[industry] = match_count
        
        # Determine the best matching industry
        if industry_matches:
            # Sort industries by match count (descending)
            sorted_industries = sorted(industry_matches.items(), key=lambda x: x[1], reverse=True)
            top_industry = sorted_industries[0][0]
            
            return {
                'sector': sector,
                'industry': top_industry,
                'industry_keywords': [ind for ind, _ in sorted_industries],
                'method': sector_result['method'],
                'confidence': sector_result.get('confidence')
            }
        else:
            # Default to "General" industry within the sector if no specific matches
            return {
                'sector': sector,
                'industry': f'General {sector}',
                'method': sector_result['method'],
                'confidence': sector_result.get('confidence')
            }
    
    def batch_classify_industry(self, texts, model_key=None):
        """
        Classify industries for a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to classify
        model_key : str, optional
            Key of model to use
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with industry classifications for each text
        """
        results = []
        for text in texts:
            result = self.classify_industry(text, model_key)
            results.append(result)
        return pd.DataFrame(results)
    
    def analyze_dataframe(self, df, text_column, model_key=None, result_prefix='industry_'):
        """
        Classify industries for texts in a DataFrame column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to classify
        text_column : str
            Name of column containing texts
        model_key : str, optional
            Key of model to use
        result_prefix : str
            Prefix for new columns with classification results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added industry classification columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Classify industries for each text
        classifications = self.batch_classify_industry(df[text_column], model_key)
        
        # Add classification columns to result DataFrame
        result_df[f'{result_prefix}sector'] = classifications['sector']
        result_df[f'{result_prefix}classification'] = classifications['industry']
        
        if 'confidence' in classifications.columns and classifications['confidence'].notna().any():
            result_df[f'{result_prefix}confidence'] = classifications['confidence']
        
        if 'industry_keywords' in classifications.columns:
            result_df[f'{result_prefix}keywords'] = classifications['industry_keywords']
            
        if 'method' in classifications.columns:
            result_df[f'{result_prefix}method'] = classifications['method']
        
        return result_df
    
    def plot_industry_distribution(self, industries, sectors=None):
        """
        Plot the distribution of industries, optionally grouped by sector
        
        Parameters:
        -----------
        industries : list or pandas.Series
            Industry classifications
        sectors : list or pandas.Series, optional
            Sector classifications (to group by sector)
        """
        if sectors is not None and len(sectors) == len(industries):
            # Create a DataFrame with both sector and industry
            df = pd.DataFrame({'sector': sectors, 'industry': industries})
            
            # Group by sector and count industries
            industry_counts = df.groupby(['sector', 'industry']).size().unstack(fill_value=0)
            
            # Create a stacked bar chart
            industry_counts.plot(kind='bar', stacked=True, figsize=(14, 8))
            plt.title('Distribution of Industries by Sector')
            plt.xlabel('Sector')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Simple industry count
            if isinstance(industries, pd.Series):
                industry_counts = industries.value_counts()
            else:
                industry_counts = pd.Series(Counter(industries))
            
            # Sort by count (descending)
            industry_counts = industry_counts.sort_values(ascending=False)
            
            # Create plot
            plt.figure(figsize=(14, 8))
            bars = sns.barplot(x=industry_counts.index, y=industry_counts.values)
            
            # Add count labels on top of bars
            for i, count in enumerate(industry_counts.values):
                bars.text(i, count + 0.5, str(count), ha='center')
            
            plt.title('Distribution of Industries')
            plt.xlabel('Industry')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('analysis', 'industry_distribution.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()


class SubIndustryClassifier(IndustryClassifier):
    """
    Class for classifying financial texts into sub-industries
    Provides even more granular classification than IndustryClassifier
    """
    
    def __init__(self, text_processor=None):
        """
        Initialize the sub-industry classifier
        
        Parameters:
        -----------
        text_processor : StockTextProcessor, optional
            Text processor to use for preprocessing. Creates a new one if None
        """
        super().__init__(text_processor)
        
        # Sub-industry groups within industries
        self.subindustry_groups = {
            # Technology sub-industries
            'Software': [
                'Enterprise Software', 'Application Software', 'Operating Systems',
                'Development Tools', 'Database Software', 'Security Software',
                'Productivity Software', 'Gaming Software', 'Mobile Apps'
            ],
            'Cloud Services': [
                'Infrastructure-as-a-Service', 'Platform-as-a-Service', 'Software-as-a-Service',
                'Data Storage', 'Cloud Security', 'Edge Computing', 'Hybrid Cloud',
                'Private Cloud', 'Public Cloud'
            ],
            'Semiconductors': [
                'Logic Chips', 'Memory Chips', 'Analog Semiconductors', 'Microprocessors',
                'Graphics Processors', 'Foundry Services', 'Chip Design', 'FPGA',
                'Power Management ICs', 'Automotive Semiconductors'
            ],
            
            # Healthcare sub-industries
            'Pharmaceuticals': [
                'Large Pharma', 'Generic Drugs', 'Specialty Pharma', 'OTC Medications',
                'Vaccines', 'Antibiotics', 'Oncology Drugs', 'CNS Drugs',
                'Cardiovascular Drugs', 'Respiratory Drugs'
            ],
            'Biotechnology': [
                'Gene Therapy', 'Immunotherapy', 'Cell Therapy', 'Genomics',
                'Proteomics', 'Biologics', 'Monoclonal Antibodies', 'RNA-based Therapies',
                'Precision Medicine', 'CRISPR & Gene Editing'
            ],
            
            # Financial Services sub-industries
            'Banks': [
                'Retail Banking', 'Commercial Banking', 'Investment Banking',
                'Online Banking', 'Community Banks', 'Global Banks',
                'Corporate Banking', 'Private Banking', 'Mortgage Banking'
            ],
            'Insurance': [
                'Life Insurance', 'Property & Casualty', 'Health Insurance',
                'Reinsurance', 'Auto Insurance', 'Home Insurance',
                'Business Insurance', 'Specialty Insurance', 'Insurance Brokers'
            ],
            
            # Add more as needed for other industries
        }
        
        # Sub-industry-specific keywords for classification
        self.subindustry_keywords = {
            # Software sub-industries
            'Enterprise Software': [
                'enterprise resource planning', 'erp', 'crm', 'customer relationship',
                'business process', 'supply chain software', 'workflow', 'corporate'
            ],
            'Application Software': [
                'application', 'app', 'software product', 'end-user', 'consumer software',
                'mobile application', 'desktop', 'cross-platform'
            ],
            'Security Software': [
                'antivirus', 'firewall', 'endpoint protection', 'cyber security software',
                'malware', 'threat detection', 'data protection', 'encryption software'
            ],
            
            # Semiconductor sub-industries
            'Logic Chips': [
                'logic', 'cpu', 'microprocessor', 'processing unit', 'compute',
                'central processing', 'instruction set', 'logic gate'
            ],
            'Memory Chips': [
                'memory', 'dram', 'nand', 'flash memory', 'storage chip', 'ram',
                'volatile memory', 'non-volatile memory', 'storage'
            ],
            'Graphics Processors': [
                'gpu', 'graphics card', 'graphics processing', 'video card', 'rendering',
                'compute card', 'display processor', 'visual computing'
            ],
            
            # Add more as needed for other sub-industries
        }
    
    def classify_subindustry(self, text, model_key=None):
        """
        Classify text into a sub-industry group
        
        Parameters:
        -----------
        text : str
            Text to classify
        model_key : str, optional
            Key of model to use
            
        Returns:
        --------
        dict
            Classification results with sector, industry, sub-industry, and confidence
        """
        # First get the industry classification
        industry_result = self.classify_industry(text, model_key)
        sector = industry_result['sector']
        industry = industry_result['industry']
        
        # Get possible sub-industries for this industry
        possible_subindustries = self.subindustry_groups.get(industry, [])
        
        if not possible_subindustries:
            # If industry has no defined sub-industries
            return {
                'sector': sector,
                'industry': industry,
                'subindustry': industry,  # Use industry name as sub-industry
                'method': industry_result['method'],
                'confidence': industry_result.get('confidence')
            }
        
        # Preprocess text
        processed_text = self.text_processor.preprocess_text(text)
        
        # Count keyword matches for each sub-industry
        subindustry_matches = {}
        
        for subindustry in possible_subindustries:
            match_count = 0
            
            # Use sub-industry keywords if available
            keywords = self.subindustry_keywords.get(subindustry, [])
            
            if not keywords:
                # If no specific keywords for this sub-industry, use the sub-industry name itself
                keywords = [subindustry.lower()]
            
            for keyword in keywords:
                # Look for the keyword as a whole word or phrase
                matches = re.findall(r'\b{}\b'.format(re.escape(keyword)), processed_text.lower())
                match_count += len(matches)
            
            if match_count > 0:
                subindustry_matches[subindustry] = match_count
        
        # Determine the best matching sub-industry
        if subindustry_matches:
            # Sort sub-industries by match count (descending)
            sorted_subindustries = sorted(subindustry_matches.items(), key=lambda x: x[1], reverse=True)
            top_subindustry = sorted_subindustries[0][0]
            
            return {
                'sector': sector,
                'industry': industry,
                'subindustry': top_subindustry,
                'subindustry_keywords': [sub for sub, _ in sorted_subindustries],
                'method': industry_result['method'],
                'confidence': industry_result.get('confidence')
            }
        else:
            # Default to industry name as sub-industry if no specific matches
            return {
                'sector': sector,
                'industry': industry,
                'subindustry': f'General {industry}',
                'method': industry_result['method'],
                'confidence': industry_result.get('confidence')
            }
    
    def batch_classify_subindustry(self, texts, model_key=None):
        """
        Classify sub-industries for a batch of texts
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to classify
        model_key : str, optional
            Key of model to use
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sub-industry classifications for each text
        """
        results = []
        for text in texts:
            result = self.classify_subindustry(text, model_key)
            results.append(result)
        return pd.DataFrame(results)
    
    def analyze_dataframe(self, df, text_column, model_key=None, result_prefix='subindustry_'):
        """
        Classify sub-industries for texts in a DataFrame column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing texts to classify
        text_column : str
            Name of column containing texts
        model_key : str, optional
            Key of model to use
        result_prefix : str
            Prefix for new columns with classification results
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added sub-industry classification columns
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Classify sub-industries for each text
        classifications = self.batch_classify_subindustry(df[text_column], model_key)
        
        # Add classification columns to result DataFrame
        result_df[f'{result_prefix}sector'] = classifications['sector']
        result_df[f'{result_prefix}industry'] = classifications['industry']
        result_df[f'{result_prefix}classification'] = classifications['subindustry']
        
        if 'confidence' in classifications.columns and classifications['confidence'].notna().any():
            result_df[f'{result_prefix}confidence'] = classifications['confidence']
        
        if 'subindustry_keywords' in classifications.columns:
            result_df[f'{result_prefix}keywords'] = classifications['subindustry_keywords']
            
        if 'method' in classifications.columns:
            result_df[f'{result_prefix}method'] = classifications['method']
        
        return result_df
    
    def plot_subindustry_distribution(self, subindustries, industries=None, sectors=None):
        """
        Plot the distribution of sub-industries, optionally grouped by industry or sector
        
        Parameters:
        -----------
        subindustries : list or pandas.Series
            Sub-industry classifications
        industries : list or pandas.Series, optional
            Industry classifications (to group by industry)
        sectors : list or pandas.Series, optional
            Sector classifications (for further grouping)
        """
        if industries is not None and len(industries) == len(subindustries):
            # Create a DataFrame with both industry and sub-industry
            if sectors is not None and len(sectors) == len(subindustries):
                df = pd.DataFrame({
                    'sector': sectors,
                    'industry': industries,
                    'subindustry': subindustries
                })
                
                # Group by sector, industry and count sub-industries
                counts = df.groupby(['sector', 'industry', 'subindustry']).size().unstack(level=2, fill_value=0)
            else:
                df = pd.DataFrame({'industry': industries, 'subindustry': subindustries})
                
                # Group by industry and count sub-industries
                counts = df.groupby(['industry', 'subindustry']).size().unstack(fill_value=0)
            
            # Create a stacked bar chart
            counts.plot(kind='bar', stacked=True, figsize=(16, 10))
            plt.title('Distribution of Sub-Industries')
            plt.xlabel('Industry')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Sub-Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Simple sub-industry count
            if isinstance(subindustries, pd.Series):
                counts = subindustries.value_counts()
            else:
                counts = pd.Series(Counter(subindustries))
            
            # Sort by count (descending)
            counts = counts.sort_values(ascending=False)
            
            # Limit to top 20 for readability
            if len(counts) > 20:
                counts = counts.head(20)
            
            # Create plot
            plt.figure(figsize=(14, 8))
            bars = sns.barplot(x=counts.index, y=counts.values)
            
            # Add count labels on top of bars
            for i, count in enumerate(counts.values):
                bars.text(i, count + 0.5, str(count), ha='center')
            
            plt.title('Distribution of Top 20 Sub-Industries')
            plt.xlabel('Sub-Industry')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join('analysis', 'subindustry_distribution.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        plt.show()

class ESGClassifier(SectorClassifier):
    """
    Classifier for Environmental, Social, and Governance (ESG) topics in financial texts
    """
    
    def __init__(self, text_processor=None):
        """
        Initialize the ESG classifier
        
        Parameters:
        -----------
        text_processor : TextProcessor, optional
            Text processor to use for preprocessing. Creates a new one if None
        """
        super().__init__(text_processor)
        
        # ESG categories
        self.esg_categories = ['Environmental', 'Social', 'Governance']
        
        # ESG topics within each category
        self.esg_topics = {
            'Environmental': [
                'Climate Change', 'Carbon Emissions', 'Renewable Energy',
                'Pollution', 'Waste Management', 'Water Usage', 'Biodiversity',
                'Resource Conservation', 'Green Buildings', 'Sustainable Products'
            ],
            'Social': [
                'Labor Practices', 'Human Rights', 'Diversity & Inclusion',
                'Community Relations', 'Product Safety', 'Data Privacy',
                'Supply Chain Management', 'Employee Wellbeing', 'Customer Satisfaction',
                'Health & Safety'
            ],
            'Governance': [
                'Board Structure', 'Executive Compensation', 'Shareholder Rights',
                'Audit & Accounting', 'Business Ethics', 'Compliance',
                'Risk Management', 'Transparency', 'Anti-corruption',
                'Lobbying & Political Contributions'
            ]
        }
        
        # Keywords for each ESG topic
        self.esg_keywords = {
            # Environmental topics
            'Climate Change': [
                'climate', 'global warming', 'greenhouse', 'carbon footprint', 'emissions reduction',
                'paris agreement', 'climate risk', 'climate action', 'global temperature'
            ],
            'Carbon Emissions': [
                'carbon', 'co2', 'emissions', 'carbon neutral', 'net zero', 'carbon offsetting',
                'scope 1', 'scope 2', 'scope 3', 'carbon intensity', 'decarbonization'
            ],
            'Renewable Energy': [
                'renewable', 'solar', 'wind', 'hydro', 'geothermal', 'biofuel',
                'clean energy', 'green energy', 'sustainable energy', 'alternative energy'
            ],
            
            # Social topics
            'Labor Practices': [
                'labor', 'worker', 'employee', 'working condition', 'fair wage',
                'collective bargaining', 'union', 'labor right', 'child labor'
            ],
            'Diversity & Inclusion': [
                'diversity', 'inclusion', 'gender', 'equality', 'equity', 'representation',
                'minority', 'women', 'lgbtq', 'disability', 'inclusive workplace'
            ],
            'Data Privacy': [
                'privacy', 'data protection', 'gdpr', 'ccpa', 'personal data',
                'data breach', 'data security', 'consent', 'information security'
            ],
            
            # Governance topics
            'Board Structure': [
                'board', 'director', 'independent director', 'board diversity',
                'board composition', 'board oversight', 'chairman', 'ceo'
            ],
            'Executive Compensation': [
                'compensation', 'executive pay', 'pay ratio', 'incentive', 'bonus',
                'stock option', 'equity award', 'remuneration', 'performance pay', 'golden parachute'
            ],
            'Shareholder Rights': [
                'shareholder', 'voting rights', 'proxy', 'activist investor', 'minority shareholder',
                'takeover defense', 'poison pill', 'staggered board', 'dual-class shares'
            ],
            'Business Ethics': [
                'ethics', 'conduct', 'integrity', 'values', 'code of conduct', 'whistleblower',
                'ethical business', 'responsible business', 'fair practice'
            ]
        }