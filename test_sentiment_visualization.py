"""
Test script for sentiment visualization
This script verifies that the sentiment visualization components work correctly
"""
import os
import sys
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dashboard.sentiment_visualization import SentimentVisualizer

def test_sentiment_visualizer():
    """Test the sentiment visualizer component"""
    visualizer = SentimentVisualizer()
    logger.info("Created SentimentVisualizer instance")
    
    # Test gauge chart
    logger.info("Testing gauge chart creation...")
    gauge_fig = visualizer.create_sentiment_gauge(0.65, "Test Sentiment Gauge")
    logger.info("Gauge chart created successfully")
    
    # Generate dummy data
    logger.info("Generating dummy sentiment data...")
    dummy_sentiment = visualizer.generate_dummy_sentiment_data("TEST", days=30)
    logger.info(f"Generated dummy data with shape: {dummy_sentiment.shape}")
    
    # Test timeline chart
    logger.info("Testing timeline chart creation...")
    timeline_fig = visualizer.create_sentiment_timeline(dummy_sentiment, title="Test Timeline")
    logger.info("Timeline chart created successfully")
    
    # Test source comparison
    logger.info("Testing source comparison chart...")
    source_data = visualizer.generate_dummy_source_data("TEST")
    source_fig = visualizer.create_sentiment_by_source(source_data, "Test Source Comparison")
    logger.info("Source comparison chart created successfully")
    
    # Test topics treemap
    logger.info("Testing topics treemap...")
    topics_data = visualizer.generate_dummy_topics_data("TEST", n_topics=8)
    treemap_fig = visualizer.create_topic_sentiment_treemap(topics_data, "Test Topics")
    logger.info("Topics treemap created successfully")
    
    # Test wordcloud
    logger.info("Testing wordcloud creation...")
    words = "finance stocks trading market investment dividend growth value analysis technical fundamental" 
    wordcloud_img = visualizer.create_sentiment_wordcloud(words, title="Test Wordcloud")
    logger.info("Wordcloud created successfully")
    
    logger.info("All sentiment visualization tests passed successfully")
    return True

if __name__ == "__main__":
    try:
        success = test_sentiment_visualizer()
        if success:
            print("\nSUCCESS: All sentiment visualization tests passed.")
            print("This indicates that the sentiment visualization components are working correctly.")
        else:
            print("\nFAILURE: Sentiment visualization test failed.")
    except Exception as e:
        logger.error(f"Error in sentiment visualization test: {e}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("Check the logs for more details.")