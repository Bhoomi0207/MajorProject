"""
Visualization utilities for Stock Dashboard
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class StockVisualizer:
    """
    Class for creating interactive stock visualizations
    """
    def __init__(self):
        """Initialize the stock visualizer"""
        # Define enhanced color schemes
        self.colors = {
            'price': '#1f77b4',  # Blue
            'volume': '#2ca02c',  # Green
            'returns': '#ff7f0e',  # Orange
            'up': '#26a69a',  # Teal
            'down': '#ef5350',  # Red
            'sma_50': '#7986cb',  # Blue-purple
            'sma_200': '#5c6bc0',  # Deeper blue-purple
            'ema_20': '#f06292',  # Pink
            'bb_upper': '#9575cd',  # Purple
            'bb_middle': '#7e57c2',  # Deeper purple
            'bb_lower': '#9575cd',  # Purple
            'prediction': '#e040fb',  # Pink
            'background': '#f8f9fa',
            'grid': '#e6e9ec',
            'macd_line': '#26a69a',  # Teal
            'signal_line': '#ef5350',  # Red
            'histogram_up': '#26a69a',  # Teal
            'histogram_down': '#ef5350'  # Red
        }
        
        # Default chart layout with improved styling
        self.default_layout = {
            'template': 'plotly_white',
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['background'],
            'xaxis': {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            },
            'legend': {
                'orientation': 'h',
                'y': 1.02,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 12}
            },
            'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40},
            'hoverlabel': {'font_size': 12},
            'font': {'family': 'Arial, sans-serif'}
        }
    
    def create_candlestick_chart(self, df, title=None, include_volume=True, 
                                include_indicators=True, include_bollinger=False):
        """
        Create an enhanced interactive candlestick chart
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
        title : str, optional
            Chart title
        include_volume : bool
            Whether to include volume subplot
        include_indicators : bool
            Whether to include technical indicators
        include_bollinger : bool
            Whether to include Bollinger Bands
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns: {required_cols}")
            return None
        
        # Prepare technical indicators if they don't exist
        if include_indicators and 'SMA_50' not in df.columns:
            try:
                # Calculate SMA indicators
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # Add EMA indicator
                df['EMA_20'] = df['Close'].ewm(span=20).mean()
            except Exception as e:
                logger.warning(f"Could not calculate technical indicators: {e}")
        
        # Calculate Bollinger Bands if they don't exist
        if include_bollinger and ('BB_upper' not in df.columns):
            try:
                # Calculate 20-day SMA and standard deviation
                sma_20 = df['Close'].rolling(window=20).mean()
                std_20 = df['Close'].rolling(window=20).std()
                
                # Calculate Bollinger Bands (20-day, 2 standard deviations)
                df['BB_upper'] = sma_20 + (std_20 * 2)
                df['BB_middle'] = sma_20
                df['BB_lower'] = sma_20 - (std_20 * 2)
            except Exception as e:
                logger.warning(f"Could not calculate Bollinger Bands: {e}")
        
        # Calculate MACD for additional indicators
        if include_indicators and 'MACD_line' not in df.columns:
            try:
                # Calculate MACD components
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                df['MACD_line'] = ema_12 - ema_26
                df['Signal_line'] = df['MACD_line'].ewm(span=9).mean()
                df['MACD_histogram'] = df['MACD_line'] - df['Signal_line']
            except Exception as e:
                logger.warning(f"Could not calculate MACD: {e}")
        
        # Create figure with secondary y-axis for volume
        if include_volume and 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                specs=[[{"secondary_y": True}], [{}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color=self.colors['up'],
                decreasing_line_color=self.colors['down'],
                increasing_fillcolor=self.colors['up'],
                decreasing_fillcolor=self.colors['down'],
                hovertemplate='<b>Date</b>: %{x}<br>' +
                             '<b>Open</b>: $%{open:.2f}<br>' +
                             '<b>High</b>: $%{high:.2f}<br>' +
                             '<b>Low</b>: $%{low:.2f}<br>' +
                             '<b>Close</b>: $%{close:.2f}<br>'
            ),
            row=1, col=1
        )
        
        # Add volume as bar chart on second row
        if include_volume and 'Volume' in df.columns:
            # Get colors for volume bars based on price movement
            colors = np.where(df['Close'] >= df['Open'], self.colors['up'], self.colors['down'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.8,
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Volume</b>: %{y:,.0f}<br>'
                ),
                row=2, col=1
            )
        
        # Add technical indicators
        if include_indicators:
            # Add 50-day SMA
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        name='50-day SMA',
                        line=dict(color=self.colors['sma_50'], width=1.5),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>SMA (50)</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
            
            # Add 200-day SMA
            if 'SMA_200' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_200'],
                        name='200-day SMA',
                        line=dict(color=self.colors['sma_200'], width=1.5),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>SMA (200)</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
            
            # Add 20-day EMA
            if 'EMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_20'],
                        name='20-day EMA',
                        line=dict(color=self.colors['ema_20'], width=1.5, dash='dot'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>EMA (20)</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if include_bollinger:
            if all(x in df.columns for x in ['BB_upper', 'BB_middle', 'BB_lower']):
                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_upper'],
                        name='Upper Bollinger Band',
                        line=dict(color=self.colors['bb_upper'], width=1.5, dash='dash'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>Upper BB</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
                
                # Middle band
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_middle'],
                        name='Middle Bollinger Band',
                        line=dict(color=self.colors['bb_middle'], width=1.5, dash='dash'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>Middle BB</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
                
                # Lower band with fill
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_lower'],
                        name='Lower Bollinger Band',
                        line=dict(color=self.colors['bb_lower'], width=1.5, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(149, 117, 205, 0.15)',
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>Lower BB</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = {
            'text': title or 'Stock Price Chart',
            'font': {'size': 18, 'color': '#444'}
        }
        
        if include_volume:
            layout['xaxis2'] = {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title': 'Date'
            }
            layout['yaxis2'] = {
                'title': 'Volume',
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            }
        
        # Apply improved layout
        fig.update_layout(**layout)
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if include_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Add range slider with better configuration
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05,
            rangebreaks=[
                # Hide weekends
                dict(bounds=["sat", "mon"])
            ]
        )
        
        # Add watermark 
        fig.add_annotation(
            text="Stock Analytics Dashboard",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                family="Arial",
                size=30,
                color="rgba(150,150,150,0.1)"
            ),
            textangle=-30
        )
        
        return fig
    
    def create_price_chart(self, df, predictions_df=None, title=None, include_indicators=True, include_volume=False):
        """
        Create an enhanced interactive line chart for stock prices
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        predictions_df : pandas.DataFrame, optional
            DataFrame with price predictions
        title : str, optional
            Chart title
        include_indicators : bool
            Whether to include technical indicators
        include_volume : bool
            Whether to include volume subplot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Check required columns
        if 'Close' not in df.columns:
            logger.error("DataFrame missing 'Close' column")
            return None
        
        # Prepare technical indicators if they don't exist
        if include_indicators and 'SMA_50' not in df.columns:
            try:
                # Calculate SMA indicators
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # Add EMA indicator
                df['EMA_20'] = df['Close'].ewm(span=20).mean()
            except Exception as e:
                logger.warning(f"Could not calculate technical indicators: {e}")
        
        # Create figure with volume subplot if requested
        if include_volume and 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                specs=[[{"secondary_y": False}], [{}]]
            )
        else:
            fig = go.Figure()
        
        # Add close price line
        if include_volume and 'Volume' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Close Price',
                    line=dict(color=self.colors['price'], width=2.5),
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Close</b>: $%{y:.2f}<br>'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Close Price',
                    line=dict(color=self.colors['price'], width=2.5),
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Close</b>: $%{y:.2f}<br>'
                )
            )
        
        # Add technical indicators
        if include_indicators:
            # Add 50-day SMA
            if 'SMA_50' in df.columns:
                if include_volume and 'Volume' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_50'],
                            name='50-day SMA',
                            line=dict(color=self.colors['sma_50'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (50)</b>: $%{y:.2f}<br>'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_50'],
                            name='50-day SMA',
                            line=dict(color=self.colors['sma_50'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (50)</b>: $%{y:.2f}<br>'
                        )
                    )
            
            # Add 200-day SMA
            if 'SMA_200' in df.columns:
                if include_volume and 'Volume' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_200'],
                            name='200-day SMA',
                            line=dict(color=self.colors['sma_200'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (200)</b>: $%{y:.2f}<br>'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_200'],
                            name='200-day SMA',
                            line=dict(color=self.colors['sma_200'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (200)</b>: $%{y:.2f}<br>'
                        )
                    )
                    
            # Add 20-day EMA
            if 'EMA_20' in df.columns:
                if include_volume and 'Volume' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['EMA_20'],
                            name='20-day EMA',
                            line=dict(color=self.colors['ema_20'], width=1.5, dash='dot'),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>EMA (20)</b>: $%{y:.2f}<br>'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['EMA_20'],
                            name='20-day EMA',
                            line=dict(color=self.colors['ema_20'], width=1.5, dash='dot'),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>EMA (20)</b>: $%{y:.2f}<br>'
                        )
                    )
        
        # Add volume bars if requested
        if include_volume and 'Volume' in df.columns:
            # Get colors for volume bars based on price changes
            df_temp = df.copy()
            df_temp['Price_Change'] = df_temp['Close'].pct_change()
            colors = np.where(df_temp['Price_Change'] >= 0, self.colors['up'], self.colors['down'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.8,
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Volume</b>: %{y:,.0f}<br>'
                ),
                row=2, col=1
            )
        
        # Add predictions if available
        if predictions_df is not None and 'Close' in predictions_df.columns:
            if include_volume and 'Volume' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df.index,
                        y=predictions_df['Close'],
                        name='Predictions',
                        line=dict(color=self.colors['prediction'], width=2.5, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=7, symbol='circle'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                    '<b>Predicted Price</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df.index,
                        y=predictions_df['Close'],
                        name='Predictions',
                        line=dict(color=self.colors['prediction'], width=2.5, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=7, symbol='circle'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                    '<b>Predicted Price</b>: $%{y:.2f}<br>'
                    )
                )
            
            # Add shaded area for prediction region
            if len(df) > 0 and len(predictions_df) > 0:
                # Add vertical line at the transition point between historical and prediction
                last_historical_date = df.index[-1]
                first_prediction_date = predictions_df.index[0]
                
                # Check if there's a gap between historical and prediction
                if last_historical_date != first_prediction_date:
                    if include_volume and 'Volume' in df.columns:
                        fig.add_vline(
                            x=last_historical_date,
                            line_width=1,
                            line_dash="dash",
                            line_color="gray",
                            row=1, col=1
                        )
                    else:
                        fig.add_vline(
                            x=last_historical_date,
                            line_width=1,
                            line_dash="dash",
                            line_color="gray"
                        )
                
                # Add shaded area for prediction region
                if include_volume and 'Volume' in df.columns:
                    fig.add_vrect(
                        x0=last_historical_date,
                        x1=predictions_df.index[-1],
                        fillcolor="rgba(128, 0, 128, 0.05)",
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )
                else:
                    fig.add_vrect(
                        x0=last_historical_date,
                        x1=predictions_df.index[-1],
                        fillcolor="rgba(128, 0, 128, 0.05)",
                        layer="below",
                        line_width=0
                    )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = {
            'text': title or 'Stock Price Chart',
            'font': {'size': 18, 'color': '#444'}
        }
        
        # Set axis titles and configure slider
        if include_volume and 'Volume' in df.columns:
            layout['xaxis2'] = {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title': 'Date'
            }
            layout['yaxis2'] = {
                'title': 'Volume',
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            }
            fig.update_layout(**layout)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            # Add range slider with better configuration
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.05,
                rangebreaks=[
                    # Hide weekends
                    dict(bounds=["sat", "mon"])
                ]
            )
        else:
            fig.update_layout(**layout)
            fig.update_yaxes(title_text="Price ($)")
            
            # Add range slider
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.05,
                rangebreaks=[
                    # Hide weekends
                    dict(bounds=["sat", "mon"])
                ]
            )
        
        # Add watermark 
        fig.add_annotation(
            text="Stock Analytics Dashboard",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                family="Arial",
                size=30,
                color="rgba(150,150,150,0.1)"
            ),
            textangle=-30
        )
        
        return fig

    def create_comparison_chart(self, stocks_data, title=None, normalize=True):
        """
        Create a comparison chart for multiple stocks
        
        Parameters:
        -----------
        stocks_data : dict
            Dictionary of DataFrames with stock data (ticker as key)
        title : str, optional
            Chart title
        normalize : bool
            Whether to normalize prices to the same starting point
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        if not stocks_data:
            return go.Figure()
        
        fig = go.Figure()
        
        # Set a broader color palette for multiple stocks
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Sort tickers to ensure consistent colors
        tickers = sorted(stocks_data.keys())
        
        for idx, ticker in enumerate(tickers):
            df = stocks_data[ticker]
            
            # Skip if no data or missing Close column
            if df is None or df.empty or 'Close' not in df.columns:
                logger.warning(f"Skipping {ticker} - No valid data")
                continue
            
            # Log the values for debugging
            logger.info(f"Plotting {ticker} with {len(df)} data points")
            
            # Normalize prices if requested
            if normalize:
                # Get the starting value, ensuring it's not NaN or 0
                start_val = df['Close'].iloc[0]
                if pd.isna(start_val) or start_val == 0:
                    # Try to find the first non-zero, non-NaN value
                    for val in df['Close']:
                        if not pd.isna(val) and val > 0:
                            start_val = val
                            break
                    if pd.isna(start_val) or start_val == 0:
                        logger.warning(f"Cannot normalize {ticker} - No valid starting price")
                        continue
                
                # Calculate normalized values (percentage of starting value)
                y_values = (df['Close'] / start_val) * 100
                
                # Create a fixed hover template string that doesn't use f-string interpolation in the hover template
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=y_values,
                        name=ticker,
                        line=dict(
                            color=color_palette[idx % len(color_palette)],
                            width=2.5
                        ),
                        hovertemplate='<b>%{x}</b><br>' + 
                                    ticker + ': %{y:.2f}% of starting value<br>' +
                                    'Starting: $' + f"{start_val:.2f}" + '<br>' +
                                    'Current: $%{customdata:.2f}<extra></extra>',
                        customdata=df['Close']
                    )
                )
            else:
                # Use actual price values
                y_values = df['Close']
                
                # Create a fixed hover template 
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=y_values,
                        name=ticker,
                        line=dict(
                            color=color_palette[idx % len(color_palette)],
                            width=2.5
                        ),
                        hovertemplate='<b>%{x}</b><br>' +
                                    ticker + ': $%{y:.2f}<extra></extra>'
                    )
                )
        
        # Update layout
        layout = self.default_layout.copy()
        layout.update({
            'title': {
                'text': title or 'Stock Price Comparison',
                'font': {'size': 18, 'color': '#444'}
            },
            'xaxis_title': 'Date',
            'yaxis_title': 'Normalized Price (%)' if normalize else 'Price ($)',
            'legend': {
                'orientation': 'h',
                'y': 1.02,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 12}
            },
            'height': 600
        })
        
        # Hide weekends for cleaner chart
        layout['xaxis'].update({
            'rangebreaks': [
                dict(bounds=["sat", "mon"])
            ]
        })
        
        fig.update_layout(**layout)
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05
        )
        
        # Add watermark 
        fig.add_annotation(
            text="Stock Analytics Dashboard",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                family="Arial",
                size=30,
                color="rgba(150,150,150,0.1)"
            ),
            textangle=-30
        )
        
        return fig
    
    def create_correlation_heatmap(self, stocks_data, title=None):
        """
        Create a correlation heatmap for multiple stocks
        
        Parameters:
        -----------
        stocks_data : dict
            Dictionary of DataFrames with stock data (ticker as key)
        title : str, optional
            Chart title
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        if not stocks_data or len(stocks_data) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Not enough data for correlation analysis",
                annotations=[{
                    'text': "Please select at least two stocks for comparison",
                    'showarrow': False,
                    'font': {'size': 16},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            return fig
        
        # Extract returns data
        returns_data = {}
        for ticker, df in stocks_data.items():
            if df is not None and not df.empty and 'Close' in df.columns and len(df) > 1:
                # Calculate daily returns
                returns_data[ticker] = df['Close'].pct_change().dropna()
        
        # Skip if not enough data
        if len(returns_data) < 2:
            fig = go.Figure()
            fig.update_layout(
                title="Not enough data for correlation analysis",
                annotations=[{
                    'text': "Please select stocks with more data points",
                    'showarrow': False,
                    'font': {'size': 16},
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5
                }]
            )
            return fig
        
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.index,
            y=corr_matrix.columns,
            zmin=-1, zmax=1,
            colorscale='RdBu',
            colorbar=dict(
                title=dict(
                    text="Correlation",
                    side="right"
                )
            ),
            hovertemplate='%{y} - %{x}<br>Correlation: %{z:.4f}<extra></extra>'
        ))
        
        # Add correlation values as text
        for i, ticker1 in enumerate(corr_matrix.index):
            for j, ticker2 in enumerate(corr_matrix.columns):
                fig.add_annotation(
                    x=ticker1,
                    y=ticker2,
                    text=f"{corr_matrix.iloc[j, i]:.2f}",
                    showarrow=False,
                    font=dict(
                        color="black" if abs(corr_matrix.iloc[j, i]) < 0.7 else "white",
                        size=10
                    )
                )
        
        # Update layout
        layout = self.default_layout.copy()
        layout.update({
            'title': {
                'text': title or 'Stock Returns Correlation Matrix',
                'font': {'size': 18, 'color': '#444'}
            },
            'height': 600,
            'margin': {'l': 60, 'r': 60, 't': 80, 'b': 80}
        })
        
        fig.update_layout(**layout)
        
        return fig