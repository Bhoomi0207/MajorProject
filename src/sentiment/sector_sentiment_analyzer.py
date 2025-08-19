"""
Sector-wise Sentiment Analysis
Integrates sector classification with sentiment analysis to provide 
sentiment analysis broken down by market sectors and industries
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from collections import defaultdict

from src.sector_analysis.sector_classifier import SectorClassifier, IndustryClassifier, SubIndustryClassifier
from src.sentiment.advanced_emotion_sentiment import EnhancedEmotionalAnalyzer
from src.sentiment.advanced_sentiment import AdvancedSentimentAnalyzer, FinBERTSentiment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SectorSentimentAnalyzer:
    """
    Combines sector classification with sentiment analysis to provide 
    sector-specific sentiment insights
    """
    
    def __init__(self, sector_classifier=None, sentiment_analyzer=None, use_enhanced=True, classification_level='sector'):
        """
        Initialize the sector sentiment analyzer
        
        Parameters:
        -----------
        sector_classifier : SectorClassifier or IndustryClassifier or SubIndustryClassifier, optional
            The classifier to use for sector classification. Creates a new one if None
        sentiment_analyzer : AdvancedSentimentAnalyzer or EnhancedEmotionalAnalyzer, optional
            The analyzer to use for sentiment analysis. Creates a new one if None
        use_enhanced : bool
            Whether to use enhanced emotional analysis
        classification_level : str
            Level of classification ('sector', 'industry', 'subindustry')
        """
        # Set up classification level and classifiers
        self.classification_level = classification_level
        
        if sector_classifier is None:
            if classification_level == 'subindustry':
                self.classifier = SubIndustryClassifier()
            elif classification_level == 'industry':
                self.classifier = IndustryClassifier()
            else:
                self.classifier = SectorClassifier()
        else:
            self.classifier = sector_classifier
            
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
            
        # Cached results
        self.results = None
        
    def analyze_texts(self, texts, classification_level=None, model_key=None, include_fine_emotions=True):
        """
        Analyze a list of texts to get sector-specific sentiment
        
        Parameters:
        -----------
        texts : list or pandas.Series
            Texts to analyze
        classification_level : str, optional
            Level of classification to use ('sector', 'industry', 'subindustry')
            If None, uses the level set at initialization
        model_key : str, optional
            Key of classification model to use
        include_fine_emotions : bool
            Whether to include fine-grained emotion analysis (if available)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector classification and sentiment analysis results
        """
        # Create DataFrame from texts
        if isinstance(texts, list):
            df = pd.DataFrame({'text': texts})
        elif isinstance(texts, pd.Series):
            df = pd.DataFrame({'text': texts})
        else:
            df = texts.copy()
            
        # Set classification level for this analysis
        level = classification_level if classification_level else self.classification_level
        
        # Perform sector classification based on level
        if level == 'subindustry' and isinstance(self.classifier, SubIndustryClassifier):
            result_df = self.classifier.analyze_dataframe(df, 'text')
            sector_col = 'subindustry_sector'
            industry_col = 'subindustry_industry'
            classification_col = 'subindustry_classification'
        elif level == 'industry' and (isinstance(self.classifier, IndustryClassifier) or 
                                      isinstance(self.classifier, SubIndustryClassifier)):
            result_df = self.classifier.analyze_dataframe(df, 'text')
            sector_col = 'industry_sector'
            industry_col = None
            classification_col = 'industry_classification'
        else:
            # Default to sector level
            result_df = self.classifier.analyze_dataframe(df, 'text', model_key=model_key)
            sector_col = 'sector_classification'
            industry_col = None
            classification_col = sector_col
            
        # Perform sentiment analysis based on analyzer type
        if isinstance(self.sentiment_analyzer, EnhancedEmotionalAnalyzer):
            result_df = self.sentiment_analyzer.analyze_dataframe_advanced(
                result_df, 'text', include_fine_emotions=include_fine_emotions
            )
            sentiment_col = 'sentiment_label'
            score_col = 'sentiment_score'
            emotions_available = True
        elif isinstance(self.sentiment_analyzer, FinBERTSentiment):
            result_df = self.sentiment_analyzer.analyze_dataframe(result_df, 'text')
            sentiment_col = 'finbert_sentiment'
            score_col = 'finbert_score'
            emotions_available = False
        else:
            # Generic advanced sentiment analyzer
            result_df = self.sentiment_analyzer.analyze_dataframe(result_df, 'text')
            sentiment_col = 'sentiment_sentiment'
            score_col = 'sentiment_score' if 'sentiment_score' in result_df.columns else 'sentiment'
            emotions_available = False
            
        # Save columns for later use
        self.sector_col = sector_col
        self.industry_col = industry_col
        self.classification_col = classification_col
        self.sentiment_col = sentiment_col
        self.score_col = score_col
        self.emotions_available = emotions_available
        
        # Cache results
        self.results = result_df
        
        return result_df
    
    def get_sector_sentiment_summary(self, sector=None, min_texts=3):
        """
        Get a summary of sentiment by sector/industry
        
        Parameters:
        -----------
        sector : str, optional
            Specific sector to analyze. If None, analyzes all sectors
        min_texts : int
            Minimum number of texts needed to include a sector in the summary
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector sentiment metrics
        """
        if self.results is None:
            logger.error("No analysis results available. Run analyze_texts first.")
            return None
            
        # Group by classification and compute sentiment metrics
        if sector:
            # Filter to specific sector if requested
            if self.industry_col and sector:
                subset = self.results[self.results[self.industry_col] == sector]
            else:
                subset = self.results[self.results[self.sector_col] == sector]
        else:
            subset = self.results
        
        # Group by the appropriate classification column
        grouped = subset.groupby(self.classification_col)
        
        # Get counts by sentiment for each sector/industry
        if isinstance(self.sentiment_analyzer, EnhancedEmotionalAnalyzer):
            # For the enhanced analyzer, get distribution of sentiment labels
            sentiment_counts = {}
            for name, group in grouped:
                # Skip if too few texts
                if len(group) < min_texts:
                    continue
                counts = group[self.sentiment_col].value_counts()
                sentiment_counts[name] = counts
            
            # Create a DataFrame from the sentiment counts
            summary_data = {}
            for sector_name, counts in sentiment_counts.items():
                row = {}
                total = len(subset[subset[self.classification_col] == sector_name])
                
                for sentiment_label in ['very_positive', 'positive', 'neutral', 'negative', 'very_negative']:
                    count = counts.get(sentiment_label, 0)
                    row[f'{sentiment_label}_count'] = count
                    row[f'{sentiment_label}_pct'] = count / total * 100 if total > 0 else 0
                
                row['total_texts'] = total
                row['avg_sentiment_score'] = subset[subset[self.classification_col] == sector_name][self.score_col].mean()
                row['sentiment_std'] = subset[subset[self.classification_col] == sector_name][self.score_col].std()
                
                # Add dominant emotions if available
                if self.emotions_available and 'dominant_emotion' in self.results.columns:
                    dominant_emotions = group['dominant_emotion'].value_counts().head(3)
                    row['top_emotions'] = ', '.join(dominant_emotions.index)
                
                summary_data[sector_name] = row
                
        else:
            # For other analyzers, compute positive/negative/neutral counts
            summary_data = {}
            for name, group in grouped:
                # Skip if too few texts
                if len(group) < min_texts:
                    continue
                
                row = {}
                total = len(group)
                
                # Count sentiments (adjust column names if needed)
                if 'finbert_sentiment' in group.columns:
                    sentiment_column = 'finbert_sentiment'
                else:
                    sentiment_column = self.sentiment_col
                
                try:
                    positive_count = (group[sentiment_column] == 'positive').sum()
                    negative_count = (group[sentiment_column] == 'negative').sum()
                    neutral_count = (group[sentiment_column] == 'neutral').sum()
                
                    row['positive_count'] = positive_count
                    row['positive_pct'] = positive_count / total * 100 if total > 0 else 0
                    row['negative_count'] = negative_count
                    row['negative_pct'] = negative_count / total * 100 if total > 0 else 0
                    row['neutral_count'] = neutral_count
                    row['neutral_pct'] = neutral_count / total * 100 if total > 0 else 0
                    row['total_texts'] = total
                    row['avg_sentiment_score'] = group[self.score_col].mean()
                    row['sentiment_std'] = group[self.score_col].std()
                    
                    # Sentiment ratio (positive to negative)
                    row['sentiment_ratio'] = positive_count / max(negative_count, 1)
                except Exception as e:
                    logger.error(f"Error calculating sentiment metrics for {name}: {e}")
                    continue
                
                summary_data[name] = row
        
        # Create summary DataFrame
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        
        # Sort by average sentiment score
        summary_df = summary_df.sort_values('avg_sentiment_score', ascending=False)
        
        return summary_df
    
    def get_sector_sentiment(self, sector):
        """
        Get time series of sentiment data for a specific sector
        
        Parameters:
        -----------
        sector : str
            The sector to get sentiment data for
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment time series data for the sector
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        if self.results is None:
            logger.warning(f"No analysis results available for sector: {sector}")
            # Generate sample data for the sector if no real data available
            return self._generate_sample_sentiment_data(sector)
        
        # Try to filter the results for the specified sector
        if self.classification_col in self.results.columns:
            sector_data = self.results[self.results[self.classification_col] == sector]
            
            if len(sector_data) > 0 and 'date' in sector_data.columns:
                # If we have real data with dates, aggregate by date
                daily_sentiment = sector_data.groupby(pd.Grouper(key='date', freq='D')).agg({
                    self.score_col: ['mean', 'count', 'std'],
                    self.sentiment_col: lambda x: (x == 'positive').mean() * 100
                })
                
                # Flatten multi-level columns
                daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
                daily_sentiment = daily_sentiment.reset_index()
                
                # Rename columns to standard names expected by the dashboard
                daily_sentiment.rename(columns={
                    f'{self.score_col}_mean': 'sentiment_score',
                    f'{self.score_col}_count': 'news_count',
                    f'{self.sentiment_col}_<lambda_0>': 'positive_pct'
                }, inplace=True)
                
                # Add social count for dashboard compatibility
                daily_sentiment['social_count'] = np.round(daily_sentiment['news_count'] * 0.6)
                
                return daily_sentiment
            
        # If we don't have real data or it doesn't have dates, generate sample data
        logger.warning(f"No time series data available for sector: {sector}, generating sample data")
        return self._generate_sample_sentiment_data(sector)
    
    def _generate_sample_sentiment_data(self, sector):
        """Generate sample sentiment data for a sector when real data is not available"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create date range for the past 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with sample data
        # Seed random generator with sector name for consistent results
        np.random.seed(hash(sector) % 10000)
        
        # Different parameters for different sectors
        sector_params = {
            'Technology': {'base': 0.3, 'trend': 0.002, 'volatility': 0.08},
            'Healthcare': {'base': 0.2, 'trend': 0.001, 'volatility': 0.06},
            'Financial Services': {'base': 0.1, 'trend': 0.0, 'volatility': 0.10},
            'Consumer Cyclical': {'base': 0.15, 'trend': -0.001, 'volatility': 0.09},
            'Energy': {'base': -0.1, 'trend': 0.003, 'volatility': 0.12},
            'Communication Services': {'base': 0.25, 'trend': 0.001, 'volatility': 0.07},
            'Consumer Defensive': {'base': 0.05, 'trend': 0.0, 'volatility': 0.05},
            'Industrials': {'base': 0.0, 'trend': 0.001, 'volatility': 0.08},
            'Basic Materials': {'base': -0.05, 'trend': 0.002, 'volatility': 0.11},
            'Real Estate': {'base': -0.15, 'trend': 0.0, 'volatility': 0.09},
            'Utilities': {'base': 0.0, 'trend': 0.0, 'volatility': 0.04},
            'Financial': {'base': 0.1, 'trend': 0.0, 'volatility': 0.10}  # Alternate name
        }
        
        # Get parameters for this sector or use defaults
        params = sector_params.get(sector, {'base': 0.0, 'trend': 0.0, 'volatility': 0.08})
        
        # Generate sentiment score with trend and noise
        time_factor = np.arange(len(date_range))
        base_sentiment = params['base'] + time_factor * params['trend']
        noise = np.random.normal(0, params['volatility'], len(date_range))
        sentiment_score = base_sentiment + noise
        
        # Ensure values are within a reasonable range
        sentiment_score = np.clip(sentiment_score, -0.8, 0.8)
        
        # Generate random counts with some correlation to sentiment
        base_count = 10 + np.random.normal(20, 5, len(date_range))
        count_factor = 1 + 0.3 * (sentiment_score - sentiment_score.mean())
        news_count = np.round(base_count * count_factor).astype(int)
        social_count = np.round(news_count * 0.6 + np.random.normal(0, 3, len(date_range))).astype(int)
        
        # Ensure counts are positive
        news_count = np.maximum(news_count, 1)
        social_count = np.maximum(social_count, 0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'sentiment_score': sentiment_score,
            'news_count': news_count,
            'social_count': social_count,
            'positive_pct': 50 + 25 * sentiment_score  # Convert to percentage scale
        })
        
        logger.info(f"Generated sample sentiment data for {sector} with {len(df)} days of data")
        return df
        
    def plot_sector_sentiment_distribution(self, sector=None, top_n=10, figsize=(14, 8), save_path=None):
        """
        Plot sentiment distribution by sector/industry
        
        Parameters:
        -----------
        sector : str, optional
            Specific sector to analyze. If None, analyzes all sectors
        top_n : int
            Number of top sectors to include in the plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        # Get summary data
        summary_df = self.get_sector_sentiment_summary(sector)
        
        if summary_df is None or summary_df.empty:
            logger.error("No data available for sector sentiment distribution plot")
            return
            
        # Limit to top N sectors by total texts
        if len(summary_df) > top_n:
            summary_df = summary_df.nlargest(top_n, 'total_texts')
            
        # Create plot
        plt.figure(figsize=figsize)
        
        # For enhanced analyzer with five sentiment categories
        if isinstance(self.sentiment_analyzer, EnhancedEmotionalAnalyzer):
            # Data for stacked bar chart
            sectors = summary_df.index
            very_pos = summary_df['very_positive_pct'] if 'very_positive_pct' in summary_df.columns else 0
            pos = summary_df['positive_pct'] if 'positive_pct' in summary_df.columns else 0
            neu = summary_df['neutral_pct'] if 'neutral_pct' in summary_df.columns else 0
            neg = summary_df['negative_pct'] if 'negative_pct' in summary_df.columns else 0
            very_neg = summary_df['very_negative_pct'] if 'very_negative_pct' in summary_df.columns else 0
            
            # Colors for sentiment categories
            colors = ['#1E8F4E', '#7ED957', '#DBDBDB', '#FF6B6B', '#D62828']
            
            # Create stacked bar chart
            bars = plt.barh(sectors, very_pos, color=colors[0], label='Very Positive')
            bars = plt.barh(sectors, pos, left=very_pos, color=colors[1], label='Positive')
            bars = plt.barh(sectors, neu, left=very_pos+pos, color=colors[2], label='Neutral')
            bars = plt.barh(sectors, neg, left=very_pos+pos+neu, color=colors[3], label='Negative')
            bars = plt.barh(sectors, very_neg, left=very_pos+pos+neu+neg, color=colors[4], label='Very Negative')
            
            plt.xlabel('Percentage (%)')
            plt.title('Sentiment Distribution by Sector/Industry')
            plt.legend(loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Add total counts as text
            for i, sector in enumerate(sectors):
                plt.text(101, i, f"n={summary_df.loc[sector, 'total_texts']}", va='center')
                
        else:
            # For other analyzers with three sentiment categories
            sectors = summary_df.index
            pos = summary_df['positive_pct'] if 'positive_pct' in summary_df.columns else 0
            neu = summary_df['neutral_pct'] if 'neutral_pct' in summary_df.columns else 0
            neg = summary_df['negative_pct'] if 'negative_pct' in summary_df.columns else 0
            
            # Colors for sentiment categories
            colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
            
            # Create stacked bar chart
            bars = plt.barh(sectors, pos, color=colors[0], label='Positive')
            bars = plt.barh(sectors, neu, left=pos, color=colors[1], label='Neutral')
            bars = plt.barh(sectors, neg, left=pos+neu, color=colors[2], label='Negative')
            
            plt.xlabel('Percentage (%)')
            plt.title('Sentiment Distribution by Sector/Industry')
            plt.legend(loc='lower right')
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Add total counts as text
            for i, sector in enumerate(sectors):
                plt.text(101, i, f"n={summary_df.loc[sector, 'total_texts']}", va='center')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_sector_sentiment_heatmap(self, metric='avg_sentiment_score', figsize=(14, 10), save_path=None):
        """
        Plot a heatmap of sentiment metrics by sector/industry
        
        Parameters:
        -----------
        metric : str
            Metric to plot ('avg_sentiment_score', 'positive_pct', 'sentiment_ratio')
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        # Get summary data
        summary_df = self.get_sector_sentiment_summary()
        
        if summary_df is None or summary_df.empty:
            logger.error("No data available for sector sentiment heatmap")
            return
            
        if metric not in summary_df.columns:
            logger.error(f"Metric {metric} not available in summary data")
            return
            
        # Select data for heatmap
        if metric == 'avg_sentiment_score':
            data = summary_df['avg_sentiment_score'].copy()
            title = 'Average Sentiment Score by Sector/Industry'
            cmap = 'RdYlGn'  # Red (negative) to Green (positive)
        elif metric == 'positive_pct':
            data = summary_df['positive_pct'].copy()
            title = 'Positive Sentiment Percentage by Sector/Industry'
            cmap = 'YlGn'  # Yellow to Green
        elif metric == 'sentiment_ratio':
            data = summary_df['sentiment_ratio'].copy()
            title = 'Positive/Negative Ratio by Sector/Industry'
            cmap = 'RdYlGn'  # Red to Green
        else:
            data = summary_df[metric].copy()
            title = f'{metric} by Sector/Industry'
            cmap = 'viridis'
            
        # Reshape data for heatmap (one row)
        data_df = pd.DataFrame(data).T
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.heatmap(data_df, annot=True, fmt='.2f', cmap=cmap, 
                        linewidths=0.5, cbar_kws={'label': metric})
        
        plt.title(title)
        plt.yticks([])  # Hide y-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def plot_sector_emotion_distribution(self, top_n=5, figsize=(14, 10), save_path=None):
        """
        Plot distribution of emotions by sector
        
        Parameters:
        -----------
        top_n : int
            Number of top sectors to include
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.emotions_available or self.results is None:
            logger.error("Emotion data not available")
            return
            
        if 'dominant_emotion' not in self.results.columns:
            logger.error("Dominant emotion data not found in results")
            return
            
        # Get counts of dominant emotions by sector
        emotion_counts = self.results.groupby([self.classification_col, 'dominant_emotion']).size().unstack(fill_value=0)
        
        # Select top N sectors by total count
        if len(emotion_counts) > top_n:
            totals = emotion_counts.sum(axis=1)
            top_sectors = totals.nlargest(top_n).index
            emotion_counts = emotion_counts.loc[top_sectors]
            
        # Create plot
        plt.figure(figsize=figsize)
        
        # Convert to percentage
        emotion_pcts = emotion_counts.div(emotion_counts.sum(axis=1), axis=0) * 100
        
        # Plot heatmap
        sns.heatmap(emotion_pcts, annot=True, fmt='.1f', cmap='viridis', 
                   linewidths=0.5, cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Dominant Emotions by Sector/Industry')
        plt.xlabel('Emotion')
        plt.ylabel('Sector/Industry')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def get_top_texts_by_sector(self, sector, sentiment='positive', n=5):
        """
        Get top texts for a sector based on sentiment score
        
        Parameters:
        -----------
        sector : str
            Sector/industry to get texts for
        sentiment : str
            Type of sentiment to find ('positive', 'negative', 'neutral')
        n : int
            Number of texts to return
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with top texts by sentiment score
        """
        if self.results is None:
            logger.error("No analysis results available. Run analyze_texts first.")
            return None
            
        # Filter to sector and sentiment
        sector_texts = self.results[self.results[self.classification_col] == sector]
        
        if sentiment == 'positive':
            filtered = sector_texts[sector_texts[self.sentiment_col].isin(['positive', 'very_positive'])]
            sorted_texts = filtered.sort_values(self.score_col, ascending=False)
        elif sentiment == 'negative':
            filtered = sector_texts[sector_texts[self.sentiment_col].isin(['negative', 'very_negative'])]
            sorted_texts = filtered.sort_values(self.score_col, ascending=True)
        else:
            filtered = sector_texts[sector_texts[self.sentiment_col] == 'neutral']
            # For neutral, sort by how close to zero the score is
            sorted_texts = filtered.iloc[filtered[self.score_col].abs().argsort()]
            
        # Get top n rows
        top_texts = sorted_texts.head(n)
        
        # Return relevant columns
        if self.emotions_available and 'dominant_emotion' in top_texts.columns:
            return top_texts[['text', self.score_col, self.sentiment_col, 'dominant_emotion']]
        else:
            return top_texts[['text', self.score_col, self.sentiment_col]]

# Example usage if the script is run directly
if __name__ == "__main__":
    # Sample financial texts
    texts = [
        "I am extremely bullish on tech stocks, especially semiconductor companies which are showing strong growth.",
        "Healthcare stocks are facing significant regulatory challenges and could underperform the market.",
        "Financial services companies reported strong quarterly earnings, particularly in the banking sector.",
        "The utilities sector offers stability but limited growth potential in the current market environment.",
        "Renewable energy stocks are gaining momentum as government policies favor clean energy solutions.",
        "Retail stocks are struggling due to inflation and changing consumer spending patterns.",
        "The transportation industry is dealing with rising fuel costs which is hurting profitability.",
        "Cloud computing companies continue to benefit from digital transformation initiatives.",
        "Pharmaceutical companies announced promising results from clinical trials, boosting investor sentiment.",
        "The real estate sector is under pressure due to rising interest rates and financing costs."
    ]
    
    # Initialize analyzer
    analyzer = SectorSentimentAnalyzer(classification_level='industry')
    
    # Analyze texts
    results = analyzer.analyze_texts(texts)
    
    # Display sector sentiment summary
    summary = analyzer.get_sector_sentiment_summary()
    print("Sector Sentiment Summary:")
    print(summary)
    
    # Create visualizations
    analyzer.plot_sector_sentiment_distribution()
    analyzer.plot_sector_sentiment_heatmap()
    
    # If emotion analysis is available
    if analyzer.emotions_available:
        analyzer.plot_sector_emotion_distribution()
    
    # Get top positive texts for a specific sector
    top_sector = summary.index[0]  # Get sector with highest avg sentiment
    top_texts = analyzer.get_top_texts_by_sector(top_sector, sentiment='positive')
    print(f"\nTop Positive Texts for {top_sector}:")
    print(top_texts['text'].tolist())