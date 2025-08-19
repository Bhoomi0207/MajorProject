"""
Portfolio Sentiment Scoring
Implements sentiment analysis for a portfolio of stocks, providing
weighted sentiment scores and analysis across different holdings
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from collections import defaultdict

from src.sentiment.advanced_emotion_sentiment import EnhancedEmotionalAnalyzer
from src.sentiment.advanced_sentiment import AdvancedSentimentAnalyzer, FinBERTSentiment
from src.sector_analysis.sector_classifier import SectorClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioSentimentScorer:
    """
    Analyzes sentiment across a portfolio of stocks with customizable weighting
    """
    
    def __init__(self, sentiment_analyzer=None, use_enhanced=True, sector_classifier=None):
        """
        Initialize the portfolio sentiment scorer
        
        Parameters:
        -----------
        sentiment_analyzer : AdvancedSentimentAnalyzer or EnhancedEmotionalAnalyzer, optional
            The analyzer to use for sentiment analysis. Creates a new one if None
        use_enhanced : bool
            Whether to use enhanced emotional analysis
        sector_classifier : SectorClassifier, optional
            Sector classifier for analyzing sector exposure and sector-based sentiment
        """
        # Set up sentiment analyzer
        if sentiment_analyzer is None:
            if use_enhanced:
                self.sentiment_analyzer = EnhancedEmotionalAnalyzer()
                try:
                    self.sentiment_analyzer.load_emotion_model()
                    self.sentiment_analyzer.load_fine_emotion_model()
                except Exception as e:
                    logger.warning(f"Could not load emotion models: {e}")
            else:
                try:
                    self.sentiment_analyzer = FinBERTSentiment()
                    self.sentiment_analyzer.load_model()
                except Exception as e:
                    logger.warning(f"Could not load FinBERT model: {e}")
                    self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        else:
            self.sentiment_analyzer = sentiment_analyzer
            
        # Set up sector classifier if provided
        self.sector_classifier = sector_classifier if sector_classifier else SectorClassifier()
        
        # Initialize portfolio data structure
        self.portfolio = {}
        self.texts_by_ticker = {}
        self.sentiment_results = {}
        self.portfolio_summary = None
        
    def add_holding(self, ticker, weight=None, name=None):
        """
        Add a stock holding to the portfolio
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        weight : float, optional
            Portfolio weight (percentage). If None, equal weighting is assumed
        name : str, optional
            Company name. If None, uses ticker as name
        """
        self.portfolio[ticker] = {
            'ticker': ticker,
            'weight': weight,
            'name': name if name else ticker,
            'texts': [],
            'sentiment_score': None,
            'sentiment_distribution': None,
            'emotions': None,
            'sector': None
        }
        
        # Initialize texts collection for this ticker
        if ticker not in self.texts_by_ticker:
            self.texts_by_ticker[ticker] = []
            
        # Try to determine sector
        try:
            sector_tickers = {}
            for sector in self.sector_classifier.default_sectors:
                tickers = self.sector_classifier.get_sector_tickers(sector)
                ticker_dict = {t[0]: t[1] for t in tickers}
                sector_tickers.update(ticker_dict)
                
                if ticker in ticker_dict:
                    self.portfolio[ticker]['sector'] = sector
                    self.portfolio[ticker]['name'] = ticker_dict[ticker]
                    break
        except Exception as e:
            logger.warning(f"Error determining sector for {ticker}: {e}")
            
    def add_texts(self, ticker, texts):
        """
        Add texts related to a specific ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        texts : list
            List of texts related to the ticker
        """
        if ticker not in self.portfolio:
            self.add_holding(ticker)
            
        # Add texts to the ticker's collection
        self.texts_by_ticker[ticker].extend(texts)
        self.portfolio[ticker]['texts'].extend(texts)
        
    def normalize_weights(self):
        """
        Normalize portfolio weights to ensure they sum to 100%
        """
        # If any weights are None, use equal weighting
        if any(info['weight'] is None for info in self.portfolio.values()):
            equal_weight = 100.0 / len(self.portfolio)
            for ticker in self.portfolio:
                self.portfolio[ticker]['weight'] = equal_weight
        else:
            # Normalize existing weights
            total_weight = sum(info['weight'] for info in self.portfolio.values())
            if total_weight > 0:
                for ticker in self.portfolio:
                    self.portfolio[ticker]['weight'] = (self.portfolio[ticker]['weight'] / total_weight) * 100.0
    
    def analyze_portfolio_sentiment(self, include_fine_emotions=True):
        """
        Analyze sentiment for all texts in the portfolio
        
        Parameters:
        -----------
        include_fine_emotions : bool
            Whether to include fine-grained emotion analysis (if available)
            
        Returns:
        --------
        dict
            Dictionary with sentiment analysis summary and metrics
        """
        # Ensure weights are normalized
        self.normalize_weights()
        
        # Analyze sentiment for each ticker
        for ticker, info in self.portfolio.items():
            texts = self.texts_by_ticker[ticker]
            
            if not texts:
                logger.warning(f"No texts to analyze for {ticker}")
                continue
                
            # Create DataFrame from texts
            df = pd.DataFrame({'text': texts})
            df['ticker'] = ticker
            
            # Perform sentiment analysis based on analyzer type
            if isinstance(self.sentiment_analyzer, EnhancedEmotionalAnalyzer):
                results = self.sentiment_analyzer.analyze_dataframe_advanced(
                    df, 'text', include_fine_emotions=include_fine_emotions
                )
                sentiment_col = 'sentiment_label'
                score_col = 'sentiment_score'
                emotions_available = True
            elif isinstance(self.sentiment_analyzer, FinBERTSentiment):
                results = self.sentiment_analyzer.analyze_dataframe(df, 'text')
                sentiment_col = 'finbert_sentiment'
                score_col = 'finbert_score'
                emotions_available = False
            else:
                # Generic advanced sentiment analyzer
                results = self.sentiment_analyzer.analyze_dataframe(df, 'text')
                sentiment_col = 'sentiment_sentiment'
                score_col = 'sentiment_score' if 'sentiment_score' in results.columns else 'sentiment'
                emotions_available = False
            
            # Store sentiment results
            self.sentiment_results[ticker] = results
            
            # Calculate sentiment metrics
            avg_score = results[score_col].mean()
            self.portfolio[ticker]['sentiment_score'] = avg_score
            
            # Calculate sentiment distribution
            sentiment_counts = results[sentiment_col].value_counts()
            sentiment_pcts = sentiment_counts / len(results) * 100
            self.portfolio[ticker]['sentiment_distribution'] = sentiment_pcts.to_dict()
            
            # Store dominant emotions if available
            if emotions_available and 'dominant_emotion' in results.columns:
                emotion_counts = results['dominant_emotion'].value_counts()
                emotion_pcts = emotion_counts / len(results) * 100
                top_emotions = emotion_pcts.head(3).to_dict()
                self.portfolio[ticker]['emotions'] = top_emotions
                
        # Calculate portfolio-level metrics
        weighted_score = 0
        count_positive = 0
        count_negative = 0
        count_neutral = 0
        ticker_scores = []
        
        for ticker, info in self.portfolio.items():
            if info['sentiment_score'] is not None:
                # Weighted score
                weight = info['weight'] / 100.0  # Convert percentage to decimal
                weighted_score += info['sentiment_score'] * weight
                ticker_scores.append((ticker, info['sentiment_score']))
                
                # Count by sentiment category
                if 'sentiment_distribution' in info and info['sentiment_distribution']:
                    dist = info['sentiment_distribution']
                    
                    # For enhanced analyzer with 5 categories
                    if isinstance(self.sentiment_analyzer, EnhancedEmotionalAnalyzer):
                        positive = dist.get('very_positive', 0) + dist.get('positive', 0)
                        negative = dist.get('very_negative', 0) + dist.get('negative', 0)
                        neutral = dist.get('neutral', 0)
                    else:
                        # For other analyzers with 3 categories
                        positive = dist.get('positive', 0)
                        negative = dist.get('negative', 0)
                        neutral = dist.get('neutral', 0)
                    
                    count_positive += positive * weight
                    count_negative += negative * weight
                    count_neutral += neutral * weight
        
        # Sort tickers by sentiment score
        ticker_scores.sort(key=lambda x: x[1], reverse=True)
        most_positive = ticker_scores[0][0] if ticker_scores else None
        most_negative = ticker_scores[-1][0] if ticker_scores else None
        
        # Create portfolio summary
        self.portfolio_summary = {
            'weighted_sentiment_score': weighted_score,
            'weighted_positive_pct': count_positive,
            'weighted_negative_pct': count_negative,
            'weighted_neutral_pct': count_neutral,
            'most_positive_ticker': most_positive,
            'most_negative_ticker': most_negative,
            'ticker_scores': dict(ticker_scores)
        }
        
        return self.portfolio_summary
    
    def get_sentiment_metrics_by_sector(self):
        """
        Get sentiment metrics grouped by market sector
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector-based sentiment metrics
        """
        if not self.portfolio_summary:
            logger.error("No portfolio analysis results. Run analyze_portfolio_sentiment first.")
            return None
            
        # Group tickers by sector
        sectors = defaultdict(list)
        for ticker, info in self.portfolio.items():
            if info['sector']:
                sectors[info['sector']].append(ticker)
                
        # Calculate metrics for each sector
        sector_data = {}
        for sector, tickers in sectors.items():
            # Skip sectors with no tickers
            if not tickers:
                continue
                
            # Calculate sector weight
            sector_weight = sum(self.portfolio[ticker]['weight'] for ticker in tickers)
            
            # Calculate weighted sentiment score for this sector
            sector_score = 0
            for ticker in tickers:
                ticker_weight = self.portfolio[ticker]['weight'] / sector_weight
                ticker_score = self.portfolio[ticker]['sentiment_score']
                if ticker_score is not None:
                    sector_score += ticker_score * ticker_weight
            
            sector_data[sector] = {
                'tickers': tickers,
                'portfolio_weight': sector_weight,
                'sentiment_score': sector_score,
                'ticker_count': len(tickers)
            }
            
        # Convert to DataFrame
        sector_df = pd.DataFrame.from_dict(sector_data, orient='index')
        
        # Sort by portfolio weight
        sector_df = sector_df.sort_values('portfolio_weight', ascending=False)
        
        return sector_df
    
    def plot_portfolio_sentiment(self, figsize=(12, 8), save_path=None):
        """
        Plot portfolio sentiment scores by ticker
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.portfolio_summary:
            logger.error("No portfolio analysis results. Run analyze_portfolio_sentiment first.")
            return
            
        ticker_scores = self.portfolio_summary['ticker_scores']
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Ticker': list(ticker_scores.keys()),
            'Sentiment Score': list(ticker_scores.values())
        })
        
        # Sort by sentiment score
        df = df.sort_values('Sentiment Score', ascending=False)
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(df['Ticker'], df['Sentiment Score'],
                      color=[self._get_color_for_score(score) for score in df['Sentiment Score']])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.ylabel('Sentiment Score')
        plt.title('Portfolio Sentiment Scores by Ticker')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_portfolio_sentiment_by_weight(self, figsize=(12, 8), save_path=None):
        """
        Plot portfolio sentiment scores by ticker, with bar width proportional to weight
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.portfolio_summary:
            logger.error("No portfolio analysis results. Run analyze_portfolio_sentiment first.")
            return
            
        # Create plot data
        tickers = []
        scores = []
        weights = []
        
        for ticker, info in self.portfolio.items():
            if info['sentiment_score'] is not None:
                tickers.append(ticker)
                scores.append(info['sentiment_score'])
                weights.append(info['weight'])
                
        # Sort by sentiment score
        sorted_data = sorted(zip(tickers, scores, weights), key=lambda x: x[1], reverse=True)
        tickers, scores, weights = zip(*sorted_data) if sorted_data else ([], [], [])
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(tickers, scores, width=[w/50 for w in weights],  # Scale weights for readability
                      color=[self._get_color_for_score(score) for score in scores])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.ylabel('Sentiment Score')
        plt.title('Portfolio Sentiment Scores by Ticker (Width = Weight)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_sentiment_distribution(self, figsize=(14, 7), save_path=None):
        """
        Plot sentiment distribution for each ticker in the portfolio
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.portfolio_summary:
            logger.error("No portfolio analysis results. Run analyze_portfolio_sentiment first.")
            return
            
        # Create figure
        plt.figure(figsize=figsize)
        
        # Determine number of categories based on analyzer
        if isinstance(self.sentiment_analyzer, EnhancedEmotionalAnalyzer):
            categories = ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']
            color_map = {
                'very_positive': '#1E8F4E',  # Dark green
                'positive': '#7ED957',       # Light green
                'neutral': '#DBDBDB',        # Gray
                'negative': '#FF6B6B',       # Light red
                'very_negative': '#D62828'   # Dark red
            }
        else:
            categories = ['positive', 'neutral', 'negative']
            color_map = {
                'positive': '#2ecc71',  # Green
                'neutral': '#3498db',   # Blue
                'negative': '#e74c3c'   # Red
            }
            
        # Get tickers and their distributions
        tickers = []
        distributions = []
        
        for ticker, info in self.portfolio.items():
            if 'sentiment_distribution' in info and info['sentiment_distribution']:
                tickers.append(ticker)
                
                # Extract distribution
                dist = info['sentiment_distribution']
                distribution = [dist.get(cat, 0) for cat in categories]
                distributions.append(distribution)
                
        if not tickers:
            logger.error("No sentiment distributions available")
            return
            
        # Create data for stacked bar chart
        data = np.array(distributions)
        
        # Create stacked bar chart
        bottom = np.zeros(len(tickers))
        
        for i, category in enumerate(categories):
            plt.bar(tickers, data[:, i], bottom=bottom, label=category.replace('_', ' ').title(),
                   color=color_map.get(category, f'C{i}'))
            bottom += data[:, i]
            
        plt.axhline(y=100, color='black', linestyle='-', alpha=0.3)
        plt.ylabel('Percentage (%)')
        plt.title('Sentiment Distribution by Ticker')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(categories))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_sector_sentiment(self, figsize=(12, 8), save_path=None):
        """
        Plot sector-based sentiment scores
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        sector_df = self.get_sentiment_metrics_by_sector()
        
        if sector_df is None or sector_df.empty:
            logger.error("No sector data available")
            return
            
        # Create plot
        plt.figure(figsize=figsize)
        
        # Create horizontal bar chart with sector weights as width
        sectors = sector_df.index
        scores = sector_df['sentiment_score']
        weights = sector_df['portfolio_weight']
        
        bars = plt.barh(sectors, scores, height=0.6, 
                       color=[self._get_color_for_score(score) for score in scores])
        
        # Add score and weight labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.05 if width >= 0 else width - 0.2, 
                    bar.get_y() + bar.get_height()/2,
                    f'Score: {width:.2f}  Weight: {weights.iloc[i]:.1f}%', 
                    va='center')
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.xlabel('Sentiment Score')
        plt.title('Portfolio Sentiment by Sector')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def _get_color_for_score(self, score):
        """
        Get color for a sentiment score
        
        Parameters:
        -----------
        score : float
            Sentiment score
            
        Returns:
        --------
        str
            Hex color code
        """
        if score >= 0.5:
            return '#1E8F4E'  # Dark green
        elif score >= 0.1:
            return '#7ED957'  # Light green
        elif score >= -0.1:
            return '#DBDBDB'  # Gray
        elif score >= -0.5:
            return '#FF6B6B'  # Light red
        else:
            return '#D62828'  # Dark red

# Example usage if the script is run directly
if __name__ == "__main__":
    # Sample texts for different stocks
    texts = {
        'AAPL': [
            "Apple's new product line looks incredibly promising.",
            "I'm bullish on AAPL because of their strong ecosystem and customer loyalty.",
            "Apple's revenue growth is slowing, but they maintain high profit margins.",
            "The new iPhone features are disappointing compared to competitors."
        ],
        'MSFT': [
            "Microsoft's cloud business continues to show impressive growth.",
            "Azure is gaining market share against AWS, boosting Microsoft's prospects.",
            "Microsoft's AI initiatives position them well for future growth."
        ],
        'GOOGL': [
            "Google's ad revenue is under pressure from regulatory concerns.",
            "Alphabet's diversification into cloud and AI provides significant upside.",
            "Google's antitrust issues could limit growth potential in the near term."
        ],
        'AMZN': [
            "Amazon is streamlining operations and improving profitability.",
            "AWS remains the leader in cloud infrastructure, driving Amazon's growth.",
            "Rising shipping costs are putting pressure on Amazon's retail margins."
        ],
        'TSLA': [
            "Tesla's production scale is improving dramatically.",
            "Competition in the EV market is intensifying, challenging Tesla's dominance.",
            "Tesla's energy business has massive long-term potential but remains small."
        ]
    }
    
    # Initialize portfolio scorer
    portfolio = PortfolioSentimentScorer()
    
    # Add holdings with example weights
    portfolio.add_holding('AAPL', weight=25)
    portfolio.add_holding('MSFT', weight=20)
    portfolio.add_holding('GOOGL', weight=15)
    portfolio.add_holding('AMZN', weight=15)
    portfolio.add_holding('TSLA', weight=10)
    
    # Add texts for each ticker
    for ticker, ticker_texts in texts.items():
        portfolio.add_texts(ticker, ticker_texts)
    
    # Analyze portfolio sentiment
    summary = portfolio.analyze_portfolio_sentiment()
    
    # Print summary
    print("Portfolio Sentiment Summary:")
    print(f"Weighted Sentiment Score: {summary['weighted_sentiment_score']:.4f}")
    print(f"Most Positive Holding: {summary['most_positive_ticker']}")
    print(f"Most Negative Holding: {summary['most_negative_ticker']}")
    
    # Display ticker sentiment scores
    print("\nTicker Sentiment Scores:")
    for ticker, score in summary['ticker_scores'].items():
        print(f"{ticker}: {score:.4f}")
    
    # Display sector metrics
    sector_df = portfolio.get_sentiment_metrics_by_sector()
    print("\nSector Sentiment Analysis:")
    print(sector_df)
    
    # Create visualizations
    portfolio.plot_portfolio_sentiment()
    portfolio.plot_portfolio_sentiment_by_weight()
    portfolio.plot_sentiment_distribution()
    portfolio.plot_sector_sentiment()