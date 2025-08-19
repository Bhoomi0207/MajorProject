#!/usr/bin/env python
"""
Enhanced Sentiment Analysis Demo

This script demonstrates how to use the enhanced emotional sentiment analyzer
to perform advanced sentiment analysis with emotional categories and intensity.
"""
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project components
from src.preprocessing.text_processor import StockTextProcessor
from src.sentiment.emotion_sentiment import EmotionalSentimentAnalyzer
from src.sentiment.advanced_emotion_sentiment import EnhancedEmotionalAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_financial_texts():
    """Load sample financial texts for sentiment analysis"""
    return [
        "I am extremely bullish on $AAPL, their new AI strategy will revolutionize the industry and drive massive growth.",
        "Markets looking very risky, I'm worried about inflation and a potential crash. Considering moving to cash.",
        "Not sure about $AMZN, their growth is slowing but AWS remains strong. Need more data for a clear picture.",
        "I'm outraged by the Fed's latest decision, this will hurt small investors while Wall Street profits!",
        "The latest earnings report exceeded expectations, shows strong guidance and healthy fundamentals.",
        "Tesla's production numbers disappointed again. Bearish on $TSLA until they solve manufacturing issues.",
        "Excited about the new product launch, could be a game-changer for the company's future revenue.",
        "Feeling uncertain about the market's direction with mixed economic signals and geopolitical tensions."
    ]

def load_market_context():
    """Load example market context for contextual analysis"""
    return {
        'market_trend': 'bullish',
        'sector': 'Technology',
        'sector_performance': 'outperforming',
        'recent_news_sentiment': 0.62,
        'volatility_index': 'moderate',
        'macro_environment': 'mixed',
        'fed_outlook': 'hawkish'
    }

def compare_sentiment_analyzers(texts):
    """
    Compare the basic and enhanced sentiment analyzers
    
    Parameters:
    -----------
    texts : list
        List of financial texts to analyze
    
    Returns:
    --------
    tuple
        (basic_results, enhanced_results) - DataFrames with analysis results
    """
    logger.info("Initializing basic emotional sentiment analyzer")
    basic_analyzer = EmotionalSentimentAnalyzer()
    
    # Try to load emotion detection model
    try:
        basic_analyzer.load_emotion_model()
    except Exception as e:
        logger.warning(f"Could not load basic emotion model: {e}")
    
    logger.info("Initializing enhanced emotional sentiment analyzer")
    enhanced_analyzer = EnhancedEmotionalAnalyzer()
    enhanced_analyzer.set_market_context(load_market_context())
    
    # Try to load emotion detection models
    try:
        enhanced_analyzer.load_emotion_model()
        enhanced_analyzer.load_fine_emotion_model()
    except Exception as e:
        logger.warning(f"Could not load enhanced emotion models: {e}")
    
    # Create input DataFrame
    df = pd.DataFrame({
        'text_id': range(1, len(texts) + 1),
        'text': texts
    })
    
    # Analyze with basic analyzer
    logger.info("Analyzing texts with basic emotional sentiment analyzer")
    basic_results = basic_analyzer.analyze_dataframe(df, 'text', 'basic_')
    
    # Analyze with enhanced analyzer
    logger.info("Analyzing texts with enhanced emotional sentiment analyzer")
    enhanced_results = enhanced_analyzer.analyze_dataframe_advanced(
        df, 'text', 'enhanced_', include_fine_emotions=True
    )
    
    # Merge results for comparison
    combined_results = pd.merge(basic_results, enhanced_results, on=['text_id', 'text'])
    
    return basic_results, enhanced_results, combined_results

def print_analysis_comparison(combined_results):
    """
    Print comparison of basic and enhanced analysis results
    
    Parameters:
    -----------
    combined_results : pandas.DataFrame
        DataFrame with combined analysis results
    """
    print("\nSENTIMENT ANALYSIS COMPARISON:")
    print("=" * 100)
    
    for _, row in combined_results.iterrows():
        print(f"\nText {row['text_id']}: {row['text']}")
        print("-" * 80)
        
        # Basic analysis results
        print("BASIC SENTIMENT ANALYSIS:")
        print(f"  Sentiment: {row['basic_label']} (Score: {row['basic_score']:.2f})")
        print(f"  Intensity: {row['basic_intensity']}")
        if 'basic_dominant_emotion' in row:
            print(f"  Dominant emotion: {row['basic_dominant_emotion']} ({row['basic_dominant_emotion_score']:.3f})")
        
        # Enhanced analysis results
        print("\nENHANCED SENTIMENT ANALYSIS:")
        print(f"  Sentiment: {row['enhanced_label']} (Score: {row['enhanced_score']:.2f})")
        print(f"  Enhanced intensity: {row['enhanced_enhanced_intensity']} (Level: {row['enhanced_intensity_level']})")
        
        if 'enhanced_dominant_emotion' in row:
            print(f"  Dominant emotion: {row['enhanced_dominant_emotion']} ({row['enhanced_dominant_emotion_score']:.3f})")
        
        if 'enhanced_fine_emotions' in row and row['enhanced_fine_emotions']:
            print(f"  Fine-grained emotions: {row['enhanced_fine_emotions']}")
        
        if 'enhanced_market_interpretation' in row:
            print(f"  Market interpretation: {row['enhanced_market_interpretation']}")
        
        if 'enhanced_market_context' in row:
            print(f"  Contextual analysis: {row['enhanced_market_context']}")
        
        print("=" * 100)

def create_comparison_visualizations(basic_results, enhanced_results, output_dir):
    """
    Create comparison visualizations between basic and enhanced analysis
    
    Parameters:
    -----------
    basic_results : pandas.DataFrame
        Results from basic sentiment analyzer
    enhanced_results : pandas.DataFrame
        Results from enhanced sentiment analyzer
    output_dir : str
        Directory to save visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compare sentiment distributions
    plt.figure(figsize=(12, 8))
    
    # Basic sentiment distribution
    plt.subplot(1, 2, 1)
    basic_counts = basic_results['basic_label'].value_counts()
    basic_counts.plot(kind='bar', color='skyblue')
    plt.title('Basic Sentiment Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Enhanced sentiment distribution
    plt.subplot(1, 2, 2)
    enhanced_counts = enhanced_results['enhanced_enhanced_intensity'].value_counts()
    # Sort by intensity level
    intensity_order = ['extremely_positive', 'very_positive', 'positive', 
                       'neutral', 'negative', 'very_negative', 'extremely_negative']
    enhanced_counts = enhanced_counts.reindex([i for i in intensity_order if i in enhanced_counts.index])
    
    # Define a color gradient
    colors = ['darkgreen', 'mediumseagreen', 'lightgreen', 'lightgrey',
              'lightcoral', 'indianred', 'darkred']
    enhanced_counts.plot(kind='bar', color=colors[:len(enhanced_counts)])
    plt.title('Enhanced Sentiment Distribution (7-point scale)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution_comparison.png'))
    
    # 2. Compare emotion detection
    plt.figure(figsize=(14, 8))
    
    # Basic emotions
    plt.subplot(1, 2, 1)
    if 'basic_dominant_emotion' in basic_results.columns:
        basic_emotions = basic_results['basic_dominant_emotion'].value_counts()
        basic_emotions.plot(kind='bar', colormap='viridis')
        plt.title('Basic Dominant Emotions')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    # Enhanced emotions (using fine emotions)
    plt.subplot(1, 2, 2)
    if 'enhanced_fine_emotions' in enhanced_results.columns:
        # Extract all emotions from the comma-separated lists
        all_emotions = []
        for emotions in enhanced_results['enhanced_fine_emotions']:
            if isinstance(emotions, str) and emotions:
                emotion_list = [e.strip() for e in emotions.split(',')]
                all_emotions.extend(emotion_list)
        
        # Count emotions
        emotion_counts = pd.Series(all_emotions).value_counts()
        # Limit to top 10 for readability
        if len(emotion_counts) > 10:
            emotion_counts = emotion_counts.head(10)
        
        emotion_counts.plot(kind='bar', colormap='plasma')
        plt.title('Enhanced Fine-Grained Emotions (Top 10)')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emotion_comparison.png'))
    
    # 3. Create emotion heatmap from enhanced results
    if 'enhanced_enhanced_intensity' in enhanced_results.columns:
        emotion_cols = [col for col in enhanced_results.columns if col.startswith('enhanced_emotion_')]
        if emotion_cols:
            plt.figure(figsize=(14, 10))
            
            # Create pivot table
            pivot_data = enhanced_results.pivot_table(
                index='enhanced_enhanced_intensity',
                values=emotion_cols,
                aggfunc='mean'
            )
            
            # Rename columns to remove the prefix
            pivot_data.columns = [col.replace('enhanced_emotion_', '') for col in pivot_data.columns]
            
            # Create heatmap
            import seaborn as sns
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis',
                       linewidths=.5, cbar_kws={'label': 'Mean Emotion Score'})
            
            plt.title('Emotion Intensity by Sentiment Category', fontsize=14)
            plt.ylabel('Sentiment Category', fontsize=12)
            plt.xlabel('Emotion', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'emotion_heatmap.png'))
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run the demo"""
    # Print intro
    print("\nENHANCED SENTIMENT ANALYSIS DEMO")
    print("================================")
    print("Comparing basic and enhanced sentiment analysis capabilities")
    
    # Load example texts
    financial_texts = load_sample_financial_texts()
    
    # Compare analyzers
    basic_results, enhanced_results, combined_results = compare_sentiment_analyzers(financial_texts)
    
    # Print detailed comparison
    print_analysis_comparison(combined_results)
    
    # Create visualizations
    output_dir = os.path.join(project_root, 'examples', 'visualizations')
    create_comparison_visualizations(basic_results, enhanced_results, output_dir)
    
    print(f"\nAnalysis complete. Visualizations saved to {output_dir}")
    
    # Return results for further analysis if needed
    return basic_results, enhanced_results, combined_results

if __name__ == "__main__":
    main()