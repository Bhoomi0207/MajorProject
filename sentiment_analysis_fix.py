"""
Sentiment Analysis Fix - Simple helper script to debug and fix sentiment visualization in the dashboard
"""
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend first

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import SentimentVisualizer
try:
    from dashboard.sentiment_visualization import SentimentVisualizer
    logger.info("Successfully imported SentimentVisualizer")
except Exception as e:
    logger.error(f"Failed to import SentimentVisualizer: {e}")
    raise

# Create a simple dashboard to test the sentiment visualization functionality
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Available stocks
available_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Create SentimentVisualizer
sentiment_visualizer = SentimentVisualizer()
logger.info("Created SentimentVisualizer instance")

# App layout
app.layout = dbc.Container([
    html.H1("Sentiment Visualization Debugging", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Select Stock", className="card-title"),
                    dcc.Dropdown(
                        id="sentiment-stock-selector",
                        options=[{'label': ticker, 'value': ticker} for ticker in available_stocks],
                        value="AAPL",
                        clearable=False
                    ),
                    html.Br(),
                    dbc.Button("Analyze Sentiment", id="sentiment-analyze-button", color="primary", className="mt-3"),
                    html.Div(id="sentiment-status", className="mt-2")
                ])
            ])
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Sentiment Gauge", className="text-center mt-4"),
            dcc.Loading(
                id="loading-gauge",
                type="circle",
                children=[
                    dcc.Graph(id="sentiment-gauge-chart", style={"height": "300px"})
                ]
            )
        ], width=6),
        dbc.Col([
            html.H4("Sentiment Timeline", className="text-center mt-4"),
            dcc.Loading(
                id="loading-timeline",
                type="circle",
                children=[
                    dcc.Graph(id="sentiment-timeline-chart", style={"height": "300px"})
                ]
            )
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Word Cloud", className="text-center mt-4"),
            dcc.Loading(
                id="loading-wordcloud",
                type="circle",
                children=[
                    dcc.Graph(id="sentiment-word-cloud", style={"height": "300px"})
                ]
            )
        ], width=12)
    ]),
])

# Callback for sentiment analysis
@app.callback(
    [Output('sentiment-gauge-chart', 'figure'),
     Output('sentiment-timeline-chart', 'figure'),
     Output('sentiment-word-cloud', 'figure'),
     Output('sentiment-status', 'children')],
    [Input('sentiment-analyze-button', 'n_clicks')],
    [State('sentiment-stock-selector', 'value')]
)
def update_sentiment(n_clicks, ticker):
    # Only update when button is clicked
    if not n_clicks:
        # Return empty figures on initial load
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[{
                'text': "Click 'Analyze Sentiment' to view chart",
                'showarrow': False,
                'font': {'size': 16},
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        return empty_fig, empty_fig, empty_fig, "Click 'Analyze Sentiment' to view results"
    
    try:
        logger.info(f"Generating sentiment analysis for {ticker}")
        
        # Generate sentiment score (a value between -1 and 1)
        sentiment_score = np.random.uniform(-0.6, 0.8)  # Random score for testing
        
        # Create gauge chart
        logger.info(f"Creating sentiment gauge chart with score: {sentiment_score}")
        gauge_fig = sentiment_visualizer.create_sentiment_gauge(
            sentiment_score, 
            title=f"Overall Sentiment for {ticker}"
        )
        
        # Generate dummy sentiment data
        logger.info("Generating dummy sentiment data")
        sentiment_df = sentiment_visualizer.generate_dummy_sentiment_data(ticker, days=30)
        
        # Create timeline chart
        logger.info("Creating timeline chart")
        timeline_fig = sentiment_visualizer.create_sentiment_timeline(
            sentiment_df,
            title=f"Sentiment Trend for {ticker}"
        )
        
        # Generate word cloud text based on ticker
        words = f"{ticker} stock market investing finance dividends growth value technical analysis"
        
        # Create word cloud
        logger.info("Creating word cloud")
        wordcloud_img = sentiment_visualizer.create_sentiment_wordcloud(
            words,
            title=f"Key Terms for {ticker}"
        )
        
        # Convert wordcloud to Plotly figure
        wordcloud_fig = go.Figure()
        wordcloud_fig.add_layout_image(
            dict(
                source=wordcloud_img,
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=1, sizey=1,
                sizing="stretch",
                layer="below"
            )
        )
        wordcloud_fig.update_layout(
            title=f"Key Terms for {ticker}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300
        )
        
        logger.info("Successfully created all visualizations")
        return gauge_fig, timeline_fig, wordcloud_fig, f"Analysis completed for {ticker}"
        
    except Exception as e:
        logger.error(f"Error in sentiment visualization: {e}", exc_info=True)
        error_fig = go.Figure()
        error_fig.update_layout(
            title="Error",
            annotations=[{
                'text': f"An error occurred: {str(e)}",
                'showarrow': False,
                'font': {'size': 16},
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5
            }]
        )
        error_message = f"Error: {str(e)}"
        return error_fig, error_fig, error_fig, error_message

if __name__ == '__main__':
    logger.info("Starting sentiment visualization debug application")
    app.run(debug=True, port=8060)