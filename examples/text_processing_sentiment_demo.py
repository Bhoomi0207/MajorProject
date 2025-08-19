#!/usr/bin/env python
"""
Text Preprocessing, Cleaning, Tokenization, and Sentiment Labeling Demo

This script demonstrates how to use the text processing and sentiment analysis 
components in the project to perform:
1. Text preprocessing and cleaning
2. Tokenization
3. Sentiment labeling using different methods
"""
import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project components
from src.preprocessing.text_processor import StockTextProcessor
from src.sentiment.basic_sentiment import BasicSentimentAnalyzer, StockMarketLexicon
try:
    from src.sentiment.advanced_sentiment import AdvancedSentimentAnalyzer, FinBERTSentiment
    has_advanced_sentiment = True
except ImportError:
    # If advanced sentiment components are not available
    has_advanced_sentiment = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Sample financial texts
    sample_texts = [
        "Apple's stock surged 5% after announcing their new AI strategy. Investors are very bullish on $AAPL's future prospects.",
        "Tesla reported a significant drop in quarterly profits. Shareholders are concerned about $TSLA's ability to meet production targets.",
        "Microsoft's cloud business continues to show strong growth, but competition is increasing in the AI space.",
        "Amazon's e-commerce division is facing challenges due to economic headwinds, however AWS remains profitable.",
        "The market is showing signs of uncertainty as inflation concerns persist. Expecting volatility in tech stocks."
    ]
    
    # Create DataFrame with sample texts
    df = pd.DataFrame({
        'id': range(1, len(sample_texts) + 1),
        'text': sample_texts
    })
    
    logger.info(f"Loaded {len(df)} sample texts")
    print("\nOriginal Texts:")
    for idx, text in enumerate(sample_texts, 1):
        print(f"{idx}. {text}")
    
    # ================================================
    # 1. TEXT PREPROCESSING AND CLEANING
    # ================================================
    logger.info("Starting text preprocessing and cleaning")
    
    # Initialize the stock text processor
    text_processor = StockTextProcessor(use_stemming=False)  # Using lemmatization instead of stemming
    
    # Clean individual text
    print("\n1. TEXT CLEANING EXAMPLE:")
    example_text = sample_texts[0]
    cleaned_text = text_processor.clean_text(
        example_text,
        remove_urls=True,
        remove_numbers=True,
        remove_special_chars=True,
        expand_contractions=True
    )
    print(f"Original: {example_text}")
    print(f"Cleaned:  {cleaned_text}")
    
    # Apply cleaning to all texts in DataFrame
    df['cleaned_text'] = df['text'].apply(lambda x: text_processor.clean_text(x))
    
    # ================================================
    # 2. TOKENIZATION
    # ================================================
    logger.info("Performing tokenization")
    
    # Tokenize example text
    print("\n2. TOKENIZATION EXAMPLE:")
    tokens = text_processor.tokenize_and_preprocess(
        cleaned_text,
        remove_stopwords=True,
        normalize_words=True,
        min_word_length=2
    )
    print(f"Tokens: {tokens}")
    
    # Apply tokenization to all texts and store token count
    df['tokens'] = df['cleaned_text'].apply(lambda x: text_processor.tokenize_and_preprocess(x))
    df['token_count'] = df['tokens'].apply(len)
    
    # Extract stock tickers
    df['stock_mentions'] = df['text'].apply(
        lambda x: ', '.join(text_processor.extract_stock_mentions(x))
    )
    
    # ================================================
    # 3. SENTIMENT ANALYSIS & LABELING
    # ================================================
    logger.info("Performing sentiment analysis using multiple methods")
    
    print("\n3. SENTIMENT ANALYSIS USING MULTIPLE METHODS:")
    
    # Method 1: Using VADER with stock-specific adjustments
    print("\nMethod 1: VADER with Stock Adjustments")
    df['vader_sentiment'] = df['cleaned_text'].apply(
        lambda x: text_processor.analyze_sentiment(x)
    )
    df['vader_label'] = df['vader_sentiment'].apply(
        lambda x: 'positive' if x['adjusted_compound'] >= 0.05 
                 else 'negative' if x['adjusted_compound'] <= -0.05 
                 else 'neutral'
    )
    df['vader_score'] = df['vader_sentiment'].apply(lambda x: x['adjusted_compound'])
    
    # Method 2: Using TextBlob
    print("Method 2: TextBlob")
    analyzer = BasicSentimentAnalyzer(text_processor=text_processor)
    textblob_results = analyzer.batch_analyze_textblob(df['cleaned_text'])
    df['textblob_label'] = textblob_results['sentiment']
    df['textblob_score'] = textblob_results['polarity']
    
    # Method 3: Using StockMarketLexicon
    print("Method 3: Stock Market Lexicon")
    lexicon = StockMarketLexicon()
    df['lexicon_results'] = df['cleaned_text'].apply(
        lambda x: lexicon.analyze_text(x, text_processor)
    )
    df['lexicon_label'] = df['lexicon_results'].apply(lambda x: x['sentiment'])
    df['lexicon_score'] = df['lexicon_results'].apply(lambda x: x['sentiment_score'])
    
    # Method 4: Using FinBERT if available
    if has_advanced_sentiment:
        print("Method 4: FinBERT")
        try:
            finbert = FinBERTSentiment(text_processor=text_processor)
            finbert_results = finbert.batch_analyze(df['cleaned_text'])
            df['finbert_label'] = finbert_results['sentiment']
            df['finbert_score'] = finbert_results['confidence']
        except Exception as e:
            logger.warning(f"Error using FinBERT: {e}")
    
    # Create a summary of sentiment analysis results
    results_df = pd.DataFrame({
        'Text': df['text'],
        'Cleaned Text': df['cleaned_text'],
        'Tokens': df['tokens'],
        'Stock Mentions': df['stock_mentions'],
        'VADER Label': df['vader_label'],
        'VADER Score': df['vader_score'],
        'TextBlob Label': df['textblob_label'],
        'TextBlob Score': df['textblob_score'],
        'Lexicon Label': df['lexicon_label'],
        'Lexicon Score': df['lexicon_score']
    })
    
    if has_advanced_sentiment and 'finbert_label' in df.columns:
        results_df['FinBERT Label'] = df['finbert_label']
        results_df['FinBERT Score'] = df['finbert_score']
    
    # Display sentiment analysis results
    print("\nSENTIMENT ANALYSIS RESULTS:")
    for idx, row in results_df.iterrows():
        print(f"\nText {idx+1}: {row['Text']}")
        print(f"Cleaned: {row['Cleaned Text']}")
        print(f"Tokens: {row['Tokens'][:10]}..." if len(row['Tokens']) > 10 else f"Tokens: {row['Tokens']}")
        print(f"Stock Mentions: {row['Stock Mentions'] or 'None'}")
        print(f"VADER: {row['VADER Label']} (Score: {row['VADER Score']:.2f})")
        print(f"TextBlob: {row['TextBlob Label']} (Score: {row['TextBlob Score']:.2f})")
        print(f"Lexicon: {row['Lexicon Label']} (Score: {row['Lexicon Score']:.2f})")
        if has_advanced_sentiment and 'FinBERT Label' in results_df.columns:
            print(f"FinBERT: {row['FinBERT Label']} (Score: {row['FinBERT Score']:.2f})")
        print("-" * 80)
    
    # Export the results to CSV
    output_path = os.path.join(project_root, 'examples', 'sentiment_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Aggregate results
    sentiment_agreement = (
        (df['vader_label'] == df['textblob_label']).sum() / len(df)
    )
    
    print(f"\nSentiment Analysis Agreement Rate (VADER vs TextBlob): {sentiment_agreement:.2%}")
    
    # Show sentiment distribution
    print("\nSentiment Distribution:")
    for method in ['vader_label', 'textblob_label', 'lexicon_label']:
        label_counts = df[method].value_counts()
        print(f"{method.replace('_label', '').capitalize()}: {dict(label_counts)}")
    
    return results_df

if __name__ == "__main__":
    main()