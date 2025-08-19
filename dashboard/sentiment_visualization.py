"""
Sentiment Visualization Module for Stock Dashboard
Provides dedicated visualizations for sentiment analysis results
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import squarify
import matplotlib.colors as mcolors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """
    Class for creating interactive sentiment visualizations
    """
    def __init__(self):
        """Initialize the sentiment visualizer"""
        # Define color schemes for sentiment
        self.colors = {
            'positive': '#26a69a',  # Teal
            'negative': '#ef5350',  # Red
            'neutral': '#7986cb',   # Blue-purple
            'background': '#f8f9fa',
            'grid': '#e6e9ec',
            'text': '#444444',
            'gauge_marker': '#1f77b4',  # Blue
        }
        
        # Sentiment color scale for continuous values
        self.sentiment_colorscale = [
            [0.0, self.colors['negative']],
            [0.5, self.colors['neutral']],
            [1.0, self.colors['positive']]
        ]
        
        # Default chart layout with improved styling
        self.default_layout = {
            'template': 'plotly_white',
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['background'],
            'font': {'family': 'Arial, sans-serif', 'color': self.colors['text']},
            'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40},
            'hoverlabel': {'font_size': 12},
        }
        # Log successful initialization
        logger.info("SentimentVisualizer initialized successfully")
    
    def create_sentiment_gauge(self, sentiment_score, title=None):
        """
        Create a gauge chart to visualize sentiment score
        
        Parameters:
        -----------
        sentiment_score : float
            The sentiment score, typically in range [-1, 1]
        title : str, optional
            Chart title
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Log the sentiment score being visualized
        logger.info(f"Creating sentiment gauge chart with score: {sentiment_score}")
        
        # Normalize sentiment score to range [0, 1]
        normalized_score = (sentiment_score + 1) / 2
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 2},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.5], 'color': self.colors['negative']},
                    {'range': [-0.5, 0.5], 'color': self.colors['neutral']},
                    {'range': [0.5, 1], 'color': self.colors['positive']}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            },
            number={
                'suffix': "",
                'prefix': "Score: ",
                'font': {'size': 24}
            },
            title={
                'text': title or "Sentiment Score",
                'font': {'size': 24}
            }
        ))
        
        # Add sentiment label
        if sentiment_score <= -0.5:
            sentiment_label = "Negative"
            color = self.colors['negative']
        elif sentiment_score <= 0.5:
            sentiment_label = "Neutral"
            color = self.colors['neutral']
        else:
            sentiment_label = "Positive"
            color = self.colors['positive']
        
        fig.add_annotation(
            text=sentiment_label,
            x=0.5,
            y=0.2,
            xref="paper",
            yref="paper",
            font=dict(size=28, color=color),
            showarrow=False
        )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['height'] = 350
        fig.update_layout(**layout)
        
        logger.info(f"Sentiment gauge chart created successfully with label: {sentiment_label}")
        return fig
    
    def create_sentiment_timeline(self, sentiment_df, price_df=None, title=None):
        """
        Create sentiment timeline chart with optional price overlay
        
        Parameters:
        -----------
        sentiment_df : pandas.DataFrame
            DataFrame with sentiment data, must contain date index and sentiment column
        price_df : pandas.DataFrame, optional
            DataFrame with price data (for overlay)
        title : str, optional
            Chart title
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Check if we have the required data
        if sentiment_df is None or sentiment_df.empty or 'sentiment' not in sentiment_df.columns:
            logger.error("Invalid sentiment data for timeline chart")
            fig = go.Figure()
            fig.update_layout(
                title="No Sentiment Data Available",
                annotations=[{
                    'text': "Sentiment data is missing or invalid",
                    'showarrow': False,
                    'font': {'size': 16},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            return fig
        
        # Log the sentiment data being visualized
        logger.info(f"Creating sentiment timeline chart with {len(sentiment_df)} data points")
        
        # Create figure with dual Y axes for price overlay
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add sentiment data with color gradient based on sentiment value
        sentiment_color = [
            self.colors['negative'] if s < -0.5 else
            (self.colors['neutral'] if s < 0.5 else self.colors['positive'])
            for s in sentiment_df['sentiment']
        ]
        
        # Add sentiment line with markers
        fig.add_trace(
            go.Scatter(
                x=sentiment_df.index,
                y=sentiment_df['sentiment'],
                mode='lines+markers',
                name='Sentiment',
                line=dict(width=2),
                marker=dict(
                    size=8,
                    color=sentiment_color,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertemplate='<b>Date</b>: %{x}<br>' +
                             '<b>Sentiment</b>: %{y:.2f}<br>'
            ),
            secondary_y=False
        )
        
        # Add sentiment moving average if we have enough data
        if len(sentiment_df) > 7:
            sentiment_df['sentiment_ma7'] = sentiment_df['sentiment'].rolling(window=7).mean()
            fig.add_trace(
                go.Scatter(
                    x=sentiment_df.index,
                    y=sentiment_df['sentiment_ma7'],
                    mode='lines',
                    name='7-Day Avg Sentiment',
                    line=dict(color='purple', width=2, dash='dash'),
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>7-Day Avg</b>: %{y:.2f}<br>'
                ),
                secondary_y=False
            )
        
        # Add horizontal line at neutral sentiment (0)
        fig.add_shape(
            type="line",
            x0=sentiment_df.index.min(),
            x1=sentiment_df.index.max(),
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dot"),
            yref="y"
        )
        
        # Add price data if provided
        if price_df is not None and 'Close' in price_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=price_df.index,
                    y=price_df['Close'],
                    mode='lines',
                    name='Stock Price',
                    line=dict(color='#ff7f0e', width=2),  # Orange
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Price</b>: $%{y:.2f}<br>'
                ),
                secondary_y=True
            )
        
        # Update axis labels
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        if price_df is not None:
            fig.update_yaxes(title_text="Price ($)", secondary_y=True)
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = title or 'Sentiment Timeline'
        layout['xaxis_title'] = 'Date'
        layout['legend'] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
        
        fig.update_layout(**layout)
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05,
            rangebreaks=[
                # Hide weekends
                dict(bounds=["sat", "mon"])
            ]
        )
        
        logger.info("Sentiment timeline chart created successfully")
        return fig
    
    def create_sentiment_wordcloud(self, text_data, sentiment_scores=None, title=None, max_words=100):
        """
        Create a word cloud visualization from text data
        
        Parameters:
        -----------
        text_data : list or str
            Text data to visualize, either a list of texts or a single string
        sentiment_scores : dict, optional
            Dictionary mapping words to their sentiment scores
        title : str, optional
            Chart title
        max_words : int
            Maximum number of words to include
            
        Returns:
        --------
        str
            Base64 encoded image of word cloud
        """
        try:
            # Log the wordcloud creation attempt
            logger.info(f"Creating sentiment wordcloud with max_words: {max_words}")
            
            # Combine all text if it's a list
            if isinstance(text_data, list):
                text = ' '.join([str(t) for t in text_data if t and not pd.isna(t)])
            else:
                text = str(text_data)
            
            # Clean text
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'[^\w\s]', '', text)
            
            # Generate word cloud
            if sentiment_scores:
                # Color words by sentiment
                def color_func(word, **kwargs):
                    # Get sentiment score or default to neutral (0)
                    score = sentiment_scores.get(word, 0)
                    # Map score from [-1, 1] to [0, 1]
                    norm_score = (score + 1) / 2
                    
                    # Generate color based on sentiment
                    if score <= -0.5:
                        return self.colors['negative']
                    elif score <= 0.5:
                        return self.colors['neutral']
                    else:
                        return self.colors['positive']
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    max_words=max_words, 
                    background_color="white",
                    color_func=color_func,
                    contour_width=1,
                    contour_color='steelblue'
                ).generate(text)
            else:
                # Default colormap
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    max_words=max_words,
                    background_color="white",
                    colormap="viridis",
                    contour_width=1,
                    contour_color='steelblue'
                ).generate(text)
            
            # Convert to image
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            if title:
                plt.title(title, fontsize=16)
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            
            # Encode as base64
            img_str = base64.b64encode(buffer.read()).decode()
            
            # Close the matplotlib figure to prevent memory leaks
            plt.close()
            
            logger.info("Wordcloud created successfully")
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error generating word cloud: {e}", exc_info=True)
            # Return a simple error message image
            plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, f"Error generating word cloud: {str(e)}", 
                    ha='center', va='center', fontsize=14)
            plt.axis("off")
            
            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            
            # Encode as base64
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            return f"data:image/png;base64,{img_str}"
    
    def create_sentiment_by_source(self, source_data, title=None):
        """
        Create a comparison dashboard for sentiment across different sources
        
        Parameters:
        -----------
        source_data : dict
            Dictionary mapping source names to their sentiment data
            Each value should be a dictionary with 'sentiment', 'volume', etc.
        title : str, optional
            Chart title
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        if not source_data:
            logger.error("No source data provided for comparison")
            fig = go.Figure()
            fig.update_layout(
                title="No Source Data Available",
                annotations=[{
                    'text': "Source comparison data is missing",
                    'showarrow': False,
                    'font': {'size': 16},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            return fig
        
        # Log the source data being visualized
        logger.info(f"Creating sentiment source comparison chart with {len(source_data)} sources")
        
        # Extract source names and sentiment values
        sources = list(source_data.keys())
        sentiment_values = [data['sentiment'] for data in source_data.values()]
        volume_values = [data.get('volume', 100) for data in source_data.values()]
        
        # Determine colors based on sentiment
        colors = [
            self.colors['negative'] if s < -0.5 else
            (self.colors['neutral'] if s < 0.5 else self.colors['positive'])
            for s in sentiment_values
        ]
        
        # Create figure with subplots for different metrics
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("Sentiment by Source", "Volume by Source")
        )
        
        # Add sentiment bar chart
        fig.add_trace(
            go.Bar(
                x=sources,
                y=sentiment_values,
                text=[f"{s:.2f}" for s in sentiment_values],
                textposition='auto',
                marker_color=colors,
                name='Sentiment',
                hovertemplate='<b>Source</b>: %{x}<br>' +
                             '<b>Sentiment</b>: %{y:.2f}<br>'
            ),
            row=1, col=1
        )
        
        # Add volume pie chart
        fig.add_trace(
            go.Pie(
                labels=sources,
                values=volume_values,
                hole=.4,
                name='Volume',
                hovertemplate='<b>Source</b>: %{label}<br>' +
                             '<b>Volume</b>: %{value}<br>' +
                             '<b>Percentage</b>: %{percent:.1%}<br>'
            ),
            row=1, col=2
        )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = title or 'Sentiment Comparison by Source'
        layout['showlegend'] = False
        layout['height'] = 500
        
        fig.update_layout(**layout)
        
        # Add horizontal reference line at 0 for sentiment
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(sources) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dot"),
            row=1, col=1
        )
        
        logger.info("Sentiment source comparison chart created successfully")
        return fig
    
    def create_topic_sentiment_treemap(self, topics_data, title=None):
        """
        Create a treemap visualization of topics and their sentiment
        
        Parameters:
        -----------
        topics_data : dict
            Dictionary mapping topics to their data including 'sentiment', 'volume', etc.
        title : str, optional
            Chart title
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        if not topics_data:
            logger.error("No topics data provided for treemap")
            fig = go.Figure()
            fig.update_layout(
                title="No Topics Data Available",
                annotations=[{
                    'text': "Topics data is missing",
                    'showarrow': False,
                    'font': {'size': 16},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            return fig
        
        # Log the topics data being visualized
        logger.info(f"Creating topic sentiment treemap with {len(topics_data)} topics")
        
        # Extract topic names, sentiment values and volume (for size)
        topics = list(topics_data.keys())
        sentiment_values = [data['sentiment'] for data in topics_data.values()]
        volume_values = [data.get('volume', 100) for data in topics_data.values()]
        
        # Map sentiment to color using a continuous scale
        normalized_sentiment = [(s + 1) / 2 for s in sentiment_values]  # Map from [-1,1] to [0,1]
        
        # Create figure with treemap
        fig = go.Figure(
            go.Treemap(
                labels=topics,
                parents=[""] * len(topics),  # All at root level
                values=volume_values,
                marker=dict(
                    colors=normalized_sentiment,
                    colorscale=self.sentiment_colorscale,
                    cmid=0.5,  # to center the color scale
                ),
                hovertemplate='<b>Topic</b>: %{label}<br>' +
                             '<b>Volume</b>: %{value}<br>' +
                             '<b>Sentiment</b>: %{customdata:.2f}<br>',
                customdata=sentiment_values,
                textinfo="label+value",
                textfont={"size": 14}
            )
        )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = title or 'Topic Sentiment Analysis'
        layout['height'] = 600
        
        fig.update_layout(**layout)
        
        # Add color scale reference
        fig.update_layout(
            coloraxis=dict(
                colorscale=self.sentiment_colorscale,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Sentiment",
                    x=1.1,
                    tickvals=[0, 0.5, 1],
                    ticktext=["Negative", "Neutral", "Positive"]
                )
            )
        )
        
        logger.info("Topic sentiment treemap created successfully")
        return fig
    
    def extract_topics(self, texts, n_topics=5, n_words=5):
        """
        Extract topics from text data using LDA
        
        Parameters:
        -----------
        texts : list
            List of text strings
        n_topics : int
            Number of topics to extract
        n_words : int
            Number of words per topic
            
        Returns:
        --------
        list
            List of topics, each with top words
        """
        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                if pd.notna(text) and text:
                    # Remove URLs
                    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
                    # Remove special characters, keep only words
                    text = re.sub(r'[^\w\s]', '', text)
                    cleaned_texts.append(text)
            
            if not cleaned_texts:
                return []
            
            # Vectorize text data
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                stop_words='english'
            )
            
            # Fit and transform
            X = vectorizer.fit_transform(cleaned_texts)
            
            # Check if we have enough data for LDA
            if X.shape[0] < 5 or X.shape[1] < n_topics:
                logger.warning("Not enough data for LDA topic extraction")
                return []
            
            # Feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=5
            ).fit(X)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-n_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'id': topic_idx,
                    'words': top_words,
                    'name': f"Topic {topic_idx+1}: {top_words[0]}"
                })
                
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []

    def generate_dummy_sentiment_data(self, ticker, days=30):
        """
        Generate dummy sentiment data for testing
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days : int
            Number of days of data to generate
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with dummy sentiment data
        """
        # Log the dummy data generation
        logger.info(f"Generating dummy sentiment data for {ticker} with {days} days")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Generate random sentiment scores with some trend and noise
        base = np.linspace(-0.5, 0.5, len(date_range))  # Base trend
        noise = np.random.normal(0, 0.3, len(date_range))  # Random noise
        sentiment = np.clip(base + noise, -1, 1)  # Clip to [-1, 1]
        
        # Create DataFrame
        df = pd.DataFrame({
            'ticker': ticker,
            'sentiment': sentiment,
            'volume': np.random.randint(50, 500, len(date_range)),
            'positive_ratio': np.clip(0.5 + sentiment/2, 0.1, 0.9),
            'negative_ratio': np.clip(0.5 - sentiment/2, 0.1, 0.9),
            'neutral_ratio': np.random.uniform(0.1, 0.3, len(date_range))
        }, index=date_range)
        
        logger.info(f"Generated dummy sentiment DataFrame with shape: {df.shape}")
        return df
    
    def generate_dummy_source_data(self, ticker, sources=None):
        """
        Generate dummy sentiment data by source for testing
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        sources : list
            List of source names, defaults to ["Twitter", "Reddit", "News", "StockTwits"]
            
        Returns:
        --------
        dict
            Dictionary with source sentiment data
        """
        if sources is None:
            sources = ["Twitter", "Reddit", "News", "StockTwits"]
        
        # Log the dummy source data generation
        logger.info(f"Generating dummy source data for {ticker} with sources: {sources}")
        
        source_data = {}
        for source in sources:
            # Generate random sentiment with source-specific bias
            if source == "Twitter":
                sentiment = np.random.uniform(-0.3, 0.7)  # Twitter tends positive
                volume = np.random.randint(300, 800)
                positive_ratio = np.random.uniform(0.4, 0.7)
            elif source == "Reddit":
                sentiment = np.random.uniform(-0.6, 0.4)  # Reddit more volatile
                volume = np.random.randint(200, 600)
                positive_ratio = np.random.uniform(0.3, 0.6)
            elif source == "News":
                sentiment = np.random.uniform(-0.2, 0.2)  # News more neutral
                volume = np.random.randint(50, 200)
                positive_ratio = np.random.uniform(0.4, 0.6)
            else:
                sentiment = np.random.uniform(-0.5, 0.5)  # Others random
                volume = np.random.randint(100, 400)
                positive_ratio = np.random.uniform(0.3, 0.7)
            
            negative_ratio = np.clip(1 - positive_ratio - np.random.uniform(0, 0.3), 0.1, 0.9)
            neutral_ratio = np.clip(1 - positive_ratio - negative_ratio, 0, 0.3)
            
            source_data[source] = {
                'ticker': ticker,
                'sentiment': sentiment,
                'volume': volume,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio
            }
        
        logger.info(f"Generated source data with {len(source_data)} sources")
        return source_data
    
    def generate_dummy_topics_data(self, ticker, n_topics=8):
        """
        Generate dummy topic data for testing
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        n_topics : int
            Number of topics to generate
            
        Returns:
        --------
        dict
            Dictionary with topic data
        """
        # Log the dummy topics generation
        logger.info(f"Generating dummy topics data for {ticker} with {n_topics} topics")
        
        # Sample stock-related topics
        topic_templates = [
            "Earnings", "Financial Results", "CEO", "Management", 
            "Product Launch", "Competition", "Market Share", "Industry Trends",
            "Regulations", "Legal Issues", "Innovation", "Technology",
            "Supply Chain", "Manufacturing", "Strategy", "Expansion",
            "Dividends", "Stock Buyback", "Valuation", "Analyst Ratings"
        ]
        
        # Randomly select topics
        selected_topics = np.random.choice(topic_templates, min(n_topics, len(topic_templates)), replace=False)
        
        topics_data = {}
        for topic in selected_topics:
            # Generate random sentiment and volume
            sentiment = np.random.uniform(-0.8, 0.8)
            volume = np.random.randint(50, 500)
            
            # For some topics, bias the sentiment
            if topic in ["Earnings", "Financial Results"]:
                sentiment = np.clip(sentiment + 0.3, -1, 1)  # Usually more positive
            elif topic in ["Legal Issues"]:
                sentiment = np.clip(sentiment - 0.4, -1, 1)  # Usually more negative
            
            topics_data[topic] = {
                'ticker': ticker,
                'sentiment': sentiment,
                'volume': volume,
                'keywords': self._generate_topic_keywords(topic)
            }
        
        logger.info(f"Generated topics data with {len(topics_data)} topics")
        return topics_data
    
    def _generate_topic_keywords(self, topic):
        """Generate dummy keywords for a topic"""
        # Define common keywords for different topics
        topic_keywords = {
            "Earnings": ["EPS", "revenue", "profit", "guidance", "quarter"],
            "Financial Results": ["growth", "margin", "performance", "forecast", "outlook"],
            "CEO": ["leadership", "executive", "vision", "strategy", "decisions"],
            "Management": ["team", "executives", "leadership", "decisions", "board"],
            "Product Launch": ["announcement", "innovation", "release", "features", "launch"],
            "Competition": ["competitor", "market", "rivalry", "advantage", "threat"],
            "Market Share": ["dominance", "growth", "competition", "industry", "position"],
            "Industry Trends": ["sector", "growth", "disruption", "innovation", "future"],
            "Regulations": ["compliance", "government", "rules", "restrictions", "policy"],
            "Legal Issues": ["lawsuit", "litigation", "settlement", "court", "dispute"],
            "Innovation": ["technology", "research", "development", "breakthrough", "patent"],
            "Technology": ["innovation", "digital", "platform", "solution", "software"],
            "Supply Chain": ["logistics", "suppliers", "inventory", "distribution", "sourcing"],
            "Manufacturing": ["production", "factory", "output", "quality", "efficiency"],
            "Strategy": ["plan", "direction", "goals", "vision", "execution"],
            "Expansion": ["growth", "market", "international", "new", "opportunity"],
            "Dividends": ["yield", "payout", "increase", "return", "shareholder"],
            "Stock Buyback": ["repurchase", "authorization", "shares", "outstanding", "EPS"],
            "Valuation": ["multiple", "PE", "overvalued", "undervalued", "fair"],
            "Analyst Ratings": ["upgrade", "downgrade", "target", "recommendation", "consensus"]
        }
        
        # Return keywords if available, otherwise generate random ones
        return topic_keywords.get(topic, [f"{topic.lower()}_{i}" for i in range(5)])