#!/usr/bin/env python
"""
Stock Data Sentiment Analysis

This script demonstrates how to process financial news/comments about Apple (AAPL)
and perform sentiment analysis on them to extract insights.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project components
from src.preprocessing.text_processor import StockTextProcessor
from src.sentiment.basic_sentiment import BasicSentimentAnalyzer, StockMarketLexicon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_financial_texts():
    """
    Load or create sample financial news/comments about Apple
    In a real application, this would load from a database or API
    """
    # Sample financial texts about Apple
    return [
        "Apple's stock has been outperforming the market since the announcement of their new AI initiatives.",
        "AAPL faces headwinds as iPhone sales decline in China market by 4% year-over-year.",
        "Investors remain bullish on $AAPL following the release of their latest financial results.",
        "The new MacBook Pro's performance is impressive, but the high price point may limit adoption.",
        "Apple's services division continues to show strong growth, offsetting weakness in hardware sales.",
        "Concerns about Apple's supply chain issues in Asia could impact future quarter earnings.",
        "Analysts at Goldman Sachs upgraded $AAPL to 'buy' with a price target of $250.",
        "Competition in the wearables market is intensifying, which could pressure Apple Watch sales.",
        "Apple's cash reserves provide a strong buffer against market volatility.",
        "The antitrust investigation into App Store practices poses regulatory risks for Apple."
    ]

def load_stock_data():
    """Load Apple stock data from CSV"""
    stock_data_path = os.path.join(project_root, 'data', 'stocks', 'AAPL_data.csv')
    try:
        df = pd.read_csv(stock_data_path, parse_dates=['Date'])
        logger.info(f"Loaded stock data with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        return None

def process_financial_texts(texts, stock_ticker='AAPL'):
    """
    Process financial texts through preprocessing, cleaning, tokenization and sentiment labeling
    
    Args:
        texts (list): List of financial texts to process
        stock_ticker (str): Stock ticker for context-specific adjustments
        
    Returns:
        pd.DataFrame: DataFrame with processed text and sentiment analysis
    """
    logger.info(f"Processing {len(texts)} financial texts about {stock_ticker}")
    
    # Create DataFrame with texts
    df = pd.DataFrame({
        'id': range(1, len(texts) + 1),
        'text': texts,
        'date': datetime.now().strftime('%Y-%m-%d')  # In real app, each text would have its own date
    })
    
    # Initialize processors
    text_processor = StockTextProcessor(use_stemming=False)
    sentiment_analyzer = BasicSentimentAnalyzer(text_processor=text_processor)
    lexicon = StockMarketLexicon()
    
    # Step 1: Text Preprocessing and Cleaning
    df['cleaned_text'] = df['text'].apply(lambda x: text_processor.clean_text(x))
    
    # Step 2: Tokenization
    df['tokens'] = df['cleaned_text'].apply(lambda x: text_processor.tokenize_and_preprocess(x))
    df['token_count'] = df['tokens'].apply(len)
    
    # Extract stock tickers mentioned
    df['stock_mentions'] = df['text'].apply(
        lambda x: ','.join(text_processor.extract_stock_mentions(x))
    )
    
    # Step 3: Sentiment Analysis with multiple methods
    
    # Method 1: VADER with stock-specific adjustments
    df['vader_sentiment'] = df['cleaned_text'].apply(
        lambda x: text_processor.analyze_sentiment(x, stock_ticker)
    )
    df['vader_label'] = df['vader_sentiment'].apply(
        lambda x: 'positive' if x['adjusted_compound'] >= 0.05 
                 else 'negative' if x['adjusted_compound'] <= -0.05 
                 else 'neutral'
    )
    df['vader_score'] = df['vader_sentiment'].apply(lambda x: x['adjusted_compound'])
    
    # Method 2: TextBlob
    textblob_results = sentiment_analyzer.batch_analyze_textblob(df['cleaned_text'])
    df['textblob_label'] = textblob_results['sentiment']
    df['textblob_score'] = textblob_results['polarity']
    
    # Method 3: Financial Lexicon
    df['lexicon_results'] = df['cleaned_text'].apply(
        lambda x: lexicon.analyze_text(x, text_processor)
    )
    df['lexicon_label'] = df['lexicon_results'].apply(lambda x: x['sentiment'])
    df['lexicon_score'] = df['lexicon_results'].apply(lambda x: x['sentiment_score'])
    df['positive_words'] = df['lexicon_results'].apply(lambda x: ', '.join(x['positive_words']))
    df['negative_words'] = df['lexicon_results'].apply(lambda x: ', '.join(x['negative_words']))
    
    # Combine sentiment scores (weighted average)
    df['combined_sentiment_score'] = (
        df['vader_score'] * 0.4 + 
        df['textblob_score'] * 0.3 + 
        df['lexicon_score'] * 0.3
    )
    
    # Overall sentiment label based on combined score
    df['sentiment_label'] = df['combined_sentiment_score'].apply(
        lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
    )
    
    # Prepare results dataframe
    results = df[['id', 'text', 'cleaned_text', 'token_count', 'stock_mentions', 
                  'vader_label', 'textblob_label', 'lexicon_label', 'sentiment_label',
                  'vader_score', 'textblob_score', 'lexicon_score', 'combined_sentiment_score',
                  'positive_words', 'negative_words']]
    
    return results

def visualize_sentiment(sentiment_df):
    """Create visualizations of sentiment analysis results"""
    # Count of sentiment labels
    plt.figure(figsize=(10, 6))
    sentiment_counts = sentiment_df['sentiment_label'].value_counts()
    
    # Plot sentiment distribution
    ax = sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Sentiment Distribution for AAPL Texts')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for i, count in enumerate(sentiment_counts):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(project_root, 'examples', 'sentiment_distribution.png')
    plt.savefig(output_path)
    logger.info(f"Saved sentiment distribution chart to {output_path}")
    
    # Create a second visualization showing sentiment scores by different methods
    plt.figure(figsize=(12, 6))
    
    sentiment_methods = ['vader_score', 'textblob_score', 'lexicon_score', 'combined_sentiment_score']
    sentiment_df[sentiment_methods].plot(kind='bar', figsize=(12, 6))
    
    plt.title('Sentiment Scores by Different Methods')
    plt.xlabel('Text ID')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.legend(labels=['VADER', 'TextBlob', 'Financial Lexicon', 'Combined'])
    
    # Save the comparison chart
    plt.tight_layout()
    output_path = os.path.join(project_root, 'examples', 'sentiment_methods_comparison.png')
    plt.savefig(output_path)
    logger.info(f"Saved sentiment methods comparison chart to {output_path}")

def main():
    # Load financial texts about Apple
    financial_texts = load_sample_financial_texts()
    
    # Process texts with sentiment analysis
    sentiment_results = process_financial_texts(financial_texts, 'AAPL')
    
    # Display results
    print("\nSENTIMENT ANALYSIS RESULTS:")
    for idx, row in sentiment_results.iterrows():
        print(f"\nText {row['id']}: {row['text']}")
        print(f"Cleaned: {row['cleaned_text']}")
        print(f"Stock Mentions: {row['stock_mentions'] or 'None'}")
        print(f"Sentiment: {row['sentiment_label'].upper()} (Score: {row['combined_sentiment_score']:.2f})")
        print(f"  - VADER: {row['vader_label']} ({row['vader_score']:.2f})")
        print(f"  - TextBlob: {row['textblob_label']} ({row['textblob_score']:.2f})")
        print(f"  - Lexicon: {row['lexicon_label']} ({row['lexicon_score']:.2f})")
        print(f"Positive words: {row['positive_words'] or 'None'}")
        print(f"Negative words: {row['negative_words'] or 'None'}")
        print("-" * 80)
    
    # Save results
    output_path = os.path.join(project_root, 'examples', 'apple_sentiment_results.csv')
    sentiment_results.to_csv(output_path, index=False)
    logger.info(f"Saved sentiment analysis results to {output_path}")
    
    # Create visualizations
    visualize_sentiment(sentiment_results)
    
    # Summary statistics
    print("\nSENTIMENT ANALYSIS SUMMARY:")
    sentiment_distribution = sentiment_results['sentiment_label'].value_counts(normalize=True) * 100
    print(f"Positive: {sentiment_distribution.get('positive', 0):.1f}%")
    print(f"Neutral: {sentiment_distribution.get('neutral', 0):.1f}%")
    print(f"Negative: {sentiment_distribution.get('negative', 0):.1f}%")
    
    # Method agreement rate
    agreement_vader_textblob = (sentiment_results['vader_label'] == sentiment_results['textblob_label']).mean() * 100
    agreement_vader_lexicon = (sentiment_results['vader_label'] == sentiment_results['lexicon_label']).mean() * 100
    agreement_textblob_lexicon = (sentiment_results['textblob_label'] == sentiment_results['lexicon_label']).mean() * 100
    
    print(f"\nMethod Agreement Rates:")
    print(f"VADER-TextBlob: {agreement_vader_textblob:.1f}%")
    print(f"VADER-Lexicon: {agreement_vader_lexicon:.1f}%")
    print(f"TextBlob-Lexicon: {agreement_textblob_lexicon:.1f}%")
    
    return sentiment_results

if __name__ == "__main__":
    main()