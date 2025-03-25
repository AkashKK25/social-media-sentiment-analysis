import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from text_preprocessing import preprocess_text, extract_entities
import glob
from datetime import datetime

def load_latest_data():
    """
    Load the most recent raw data file
    
    Returns:
        DataFrame with raw tweet data
    """
    # Get list of raw data files
    data_files = glob.glob(os.path.join('data', 'raw', 'tweets_*.csv'))
    
    # If no files found, raise error
    if not data_files:
        raise FileNotFoundError("No raw data files found. Run scrape_tweets.py first.")
    
    # Sort by filename (date)
    latest_file = sorted(data_files)[-1]
    print(f"Loading data from {latest_file}")
    
    # Load the data
    df = pd.read_csv(latest_file)
    return df

def preprocess_tweets(df):
    """
    Apply preprocessing to tweet data
    
    Args:
        df: DataFrame with raw tweet data
        
    Returns:
        DataFrame with preprocessed data
    """
    print("Preprocessing tweets...")
    
    # Copy the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Check if created_at is already datetime type
    if 'created_at' in processed_df.columns:
        print(f"created_at column type: {processed_df['created_at'].dtype}")
        
        # Convert created_at to strings first for consistent parsing
        processed_df['created_at'] = processed_df['created_at'].astype(str)
        
        try:
            # Convert to datetime with UTC=True to handle timezone consistently
            processed_df['created_at'] = pd.to_datetime(processed_df['created_at'], utc=True, errors='coerce')
            
            # Check for any NaT values that couldn't be parsed
            nat_count = processed_df['created_at'].isna().sum()
            if nat_count > 0:
                print(f"Warning: {nat_count} dates couldn't be parsed and were set to NaT")
                # Set NaT values to current time
                processed_df.loc[processed_df['created_at'].isna(), 'created_at'] = pd.Timestamp.now(tz='UTC')
            
            # Create separate date columns that don't depend on .dt accessor
            processed_df['date'] = processed_df['created_at'].dt.strftime('%Y-%m-%d')
            processed_df['hour'] = processed_df['created_at'].dt.hour
            processed_df['day_of_week'] = processed_df['created_at'].dt.day_name()
            
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            print("Creating manual date columns instead...")
            
            # Create dummy date columns
            current_time = pd.Timestamp.now(tz='UTC')
            processed_df['created_at'] = current_time
            processed_df['date'] = current_time.strftime('%Y-%m-%d')
            processed_df['hour'] = current_time.hour
            processed_df['day_of_week'] = current_time.day_name()
    else:
        print("Warning: 'created_at' column not found in data")
        # Create dummy date columns
        current_time = pd.Timestamp.now(tz='UTC')
        processed_df['created_at'] = current_time
        processed_df['date'] = current_time.strftime('%Y-%m-%d')
        processed_df['hour'] = current_time.hour
        processed_df['day_of_week'] = current_time.day_name()
    
    # Custom stopwords relevant to our domain
    custom_stopwords = [
        'iphone', 'apple', 'samsung', 'galaxy', 'phone', 'smartphone',
        'via', 'rt', 'amp', 'https', 'http', 'co', 't', 's', 'm'
    ]
    
    # Apply text preprocessing
    processed_df['cleaned_text'] = processed_df['text'].apply(
        lambda x: preprocess_text(x, custom_stopwords=custom_stopwords)
    )
    
    # Create a column with text that keeps stopwords (for readability)
    processed_df['readable_text'] = processed_df['text'].apply(
        lambda x: preprocess_text(x, remove_stops=False, lemmatize=False)
    )
    
    # Extract entities (can be computationally intensive)
    print("Extracting entities from a sample of tweets...")
    sample_size = len(processed_df)
    processed_df.loc[:sample_size, 'entities'] = processed_df['text'][:sample_size].apply(extract_entities)
    
    # Calculate text length
    processed_df['text_length'] = processed_df['text'].apply(len)
    processed_df['word_count'] = processed_df['cleaned_text'].apply(lambda x: len(x.split()))
    
    # Keep track of preprocessing date
    processed_df['processed_date'] = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Preprocessing complete. {len(processed_df)} tweets processed.")
    return processed_df

def main():
    """
    Main function to preprocess the collected tweet data
    """
    # Create processed data directory if it doesn't exist
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    
    # Load the raw data
    df = load_latest_data()
    
    # Preprocess the data
    processed_df = preprocess_tweets(df)
    
    # Get current date for filename
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Save to CSV
    output_path = os.path.join('data', 'processed', f'processed_tweets_{current_date}.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main()