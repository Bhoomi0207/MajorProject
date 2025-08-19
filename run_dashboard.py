"""
Stock Dashboard Launcher
Launch the stock visualization dashboard
"""
import os
import sys
import webbrowser
from threading import Timer
import subprocess
import logging
import importlib.util
import platform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'dash', 'dash_bootstrap_components', 'plotly', 'pandas', 'numpy',
        'matplotlib', 'wordcloud', 'scikit-learn', 'tensorflow', 'torch'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"Package {package} is installed")
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"The following packages are missing and may cause issues: {', '.join(missing)}")
        return False
    
    # Check if matplotlib backend is set properly
    if platform.system() != 'Windows':
        try:
            import matplotlib
            matplotlib.use('Agg')  # Set non-interactive backend
            logger.info("Set matplotlib backend to Agg for non-interactive usage")
        except Exception as e:
            logger.warning(f"Failed to set matplotlib backend: {e}")
    
    return True

def open_browser(port=8050):
    """Open browser to the dashboard"""
    webbrowser.open_new(f"http://localhost:{port}")

def main():
    """Launch the dashboard"""
    # Check dependencies first
    check_dependencies()
    
    # Get the absolute path to the app.py file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
    
    if not os.path.exists(app_path):
        logger.error(f"Dashboard app file not found at {app_path}")
        return
    
    # Ensure the sentiment_visualization.py file exists
    sentiment_viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "sentiment_visualization.py")
    if not os.path.exists(sentiment_viz_path):
        logger.error(f"Sentiment visualization module not found at {sentiment_viz_path}")
        return
    
    port = 8050
    
    # Set environment variables to avoid common issues
    os.environ['PYTHONUNBUFFERED'] = '1'  # Ensure output is not buffered
    
    # Open browser after a short delay
    Timer(3, open_browser, kwargs={"port": port}).start()
    
    # Run the dashboard application
    logger.info(f"Starting dashboard at http://localhost:{port}")
    try:
        subprocess.run([sys.executable, app_path])
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        print(f"Error: {e}")
        print("Check that all dependencies are installed and try again.")

if __name__ == "__main__":
    main()