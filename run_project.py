import os
import sys
import subprocess
import time
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command and display output"""
    print(f"\n{'=' * 80}")
    print(f"STEP: {description}")
    print(f"{'=' * 80}\n")
    
    # Run the command and capture output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for process to complete
    process.wait()
    
    # Check if successful
    if process.returncode != 0:
        print(f"\nERROR: {description} failed with return code {process.returncode}")
        return False
    
    print(f"\nSUCCESS: {description} completed\n")
    return True

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/results",
        "models",
        "notebooks",
        "dashboard"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("All necessary directories have been created.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the entire sentiment analysis pipeline or specific steps')
    parser.add_argument('--skip-nltk', action='store_true', help='Skip NLTK setup')
    parser.add_argument('--skip-tweets', action='store_true', help='Skip tweet collection')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--skip-topics', action='store_true', help='Skip topic modeling')
    parser.add_argument('--tweets-only', action='store_true', help='Tweet collection only')
    parser.add_argument('--dashboard-only', action='store_true', help='Run only the dashboard')
    parser.add_argument('--sample-size', type=int, default=1, 
                        help='Number of times to run the tweet collection to accumulate data')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    print(f"Starting sentiment analysis pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure directories exist
    ensure_directories()
    
    # Run setup script for NLTK data
    if not args.skip_nltk and not args.dashboard_only and not args.tweets_only:
        # Try to run the force_download_nltk.py script if it exists, otherwise use alternative approach
        if os.path.exists("scripts/force_download_nltk.py"):
            if not run_command("python scripts/force_download_nltk.py", "Setting up NLTK data"):
                print("Failed to set up NLTK data. Trying alternative approach...")
                # Create a script to download NLTK data
                nltk_script = """
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
print("NLTK data downloaded successfully.")
                """
                with open("download_nltk.py", "w") as f:
                    f.write(nltk_script)
                
                run_command("python download_nltk.py", "Downloading NLTK data (alternative method)")
        else:
            # Create and run a simple script
            nltk_script = """
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
print("NLTK data downloaded successfully.")
            """
            with open("download_nltk.py", "w") as f:
                f.write(nltk_script)
            
            run_command("python download_nltk.py", "Downloading NLTK data")
    
    # Run tweet collection multiple times to accumulate data
    if not args.skip_tweets and not args.dashboard_only:
        for i in range(args.sample_size):
            print(f"\nCollecting tweets (Run {i+1}/{args.sample_size})...")
            if not run_command("python scripts/scrape_tweets.py", f"Collecting tweets (Run {i+1}/{args.sample_size})"):
                print("Warning: Tweet collection failed but continuing with pipeline...")
            
            # Small delay between runs
            if i < args.sample_size - 1:
                print("Waiting 5 seconds before next collection...")
                time.sleep(5)
    
    # Run preprocessing
    if not args.skip_preprocess and not args.dashboard_only and not args.tweets_only:
        if not run_command("python scripts/preprocess_data.py", "Preprocessing tweets"):
            print("Error: Preprocessing failed. Cannot continue with sentiment analysis.")
            if not args.skip_sentiment and not args.skip_topics:
                sys.exit(1)
    
    # Run sentiment analysis
    if not args.skip_sentiment and not args.dashboard_only and not args.tweets_only:
        if not run_command("python scripts/run_sentiment_analysis.py", "Running sentiment analysis"):
            print("Error: Sentiment analysis failed. Cannot continue with topic modeling.")
            if not args.skip_topics:
                sys.exit(1)
    
    # Run topic modeling
    if not args.skip_topics and not args.dashboard_only and not args.tweets_only:
        if not run_command("python scripts/run_topic_modeling.py", "Running topic modeling"):
            print("Warning: Topic modeling failed but continuing to dashboard...")
    
    # Run dashboard
    if not args.tweets_only:
        if run_command("cd dashboard && streamlit run app.py", "Starting dashboard"):
            print("\nDashboard is running! Access it in your web browser.")
        else:
            print("\nError: Failed to start dashboard.")
    
    # Calculate and display total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    print(f"Pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()