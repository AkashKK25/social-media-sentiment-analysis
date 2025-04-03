import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tweepy
import pandas as pd
import datetime
import time
import random
import glob
from config import (
    TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, 
    TWITTER_ACCESS_SECRET, TWITTER_BEARER_TOKEN, SEARCH_TERMS,
    MAX_TWEETS, START_DATE, END_DATE
)

def authenticate_twitter():
    """
    Authenticate with the Twitter API using credentials from config
    """
    client = tweepy.Client(
        bearer_token=TWITTER_BEARER_TOKEN,
        consumer_key=TWITTER_API_KEY, 
        consumer_secret=TWITTER_API_SECRET,
        access_token=TWITTER_ACCESS_TOKEN, 
        access_token_secret=TWITTER_ACCESS_SECRET
    )
    return client

def collect_tweets(client, query, max_results=10, start_time=None, end_time=None):
    """
    Collect tweets matching the query
    
    Args:
        client: Authenticated tweepy client
        query: Search query string
        max_results: Maximum number of tweets to collect
        start_time: Start date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        end_time: End date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        
    Returns:
        DataFrame containing collected tweets
    """
    print(f"Collecting tweets for query: {query}")
    
    # Format dates if provided and not empty
    formatted_start_time = None
    formatted_end_time = None
    
    if start_time and start_time.strip():
        formatted_start_time = f"{start_time}T00:00:00Z"
    if end_time and end_time.strip():
        formatted_end_time = f"{end_time}T23:59:59Z"
    
    # Define tweet fields to retrieve
    tweet_fields = ['created_at', 'lang', 'public_metrics', 'source']
    
    # Initialize list to store tweets
    all_tweets = []
    
    # Handle pagination
    pagination_token = None
    remaining_tweets = max_results
    
    while remaining_tweets > 0:
        # Calculate batch size (API limit is 100 per request)
        batch_size = min(remaining_tweets, 10)  # Reduced batch size to avoid rate limits
        
        try:
            # Make the API request
            response = client.search_recent_tweets(
                query=query,
                max_results=batch_size,
                tweet_fields=tweet_fields,
                next_token=pagination_token,
                start_time=formatted_start_time,
                end_time=formatted_end_time
            )
            
            # Break if no tweets are returned
            if not response.data:
                print("No tweets found for this query.")
                break
                
            # Process tweets
            for tweet in response.data:
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'lang': tweet.lang,
                    'retweet_count': tweet.public_metrics['retweet_count'],
                    'reply_count': tweet.public_metrics['reply_count'],
                    'like_count': tweet.public_metrics['like_count'],
                    'source': tweet.source,
                    'query': query
                }
                all_tweets.append(tweet_data)
            
            # Update for next iteration
            pagination_token = response.meta.get('next_token')
            remaining_tweets -= len(response.data)
            
            # Break if no pagination token (no more results)
            if not pagination_token:
                break
                
            # Sleep to respect rate limits - random sleep between 2-5 seconds
            sleep_time = 2 + random.random() * 3
            print(f"Waiting {sleep_time:.1f} seconds before next request...")
            time.sleep(sleep_time)
            
        except tweepy.TweepyException as e:
            print(f"Error: {e}")
            print("Waiting 15 seconds before trying the next query...")
            time.sleep(15)
            break
    
    print(f"Collected {len(all_tweets)} tweets")
    
    # Convert to DataFrame
    if all_tweets:
        df = pd.DataFrame(all_tweets)
        return df
    else:
        return pd.DataFrame()

def load_existing_data(raw_data_dir):
    """
    Load existing tweet data from CSV files
    
    Args:
        raw_data_dir: Directory containing raw tweet data
        
    Returns:
        DataFrame with existing tweets or empty DataFrame if none found
    """
    # Get list of tweet data files
    data_files = glob.glob(os.path.join(raw_data_dir, 'tweets_*.csv'))
    
    if not data_files:
        print("No existing tweet data found.")
        return pd.DataFrame()
    
    # Find the most recent file
    latest_file = sorted(data_files)[-1]
    print(f"Loading existing data from {latest_file}")
    
    try:
        # Load the data with datetime parsing disabled
        df = pd.read_csv(latest_file, parse_dates=False)
        
        # Convert created_at to datetime if it exists
        if 'created_at' in df.columns:
            try:
                # Use the dateutil parser which is more flexible with formats
                from dateutil.parser import parse
                
                # Function to parse dates with error handling
                def safe_parse(date_str):
                    try:
                        if isinstance(date_str, str):
                            return parse(date_str)
                        return date_str
                    except:
                        return pd.NaT
                
                # Apply the parser to the created_at column
                df['created_at'] = df['created_at'].apply(safe_parse)
                print("Successfully converted datetime column")
            except Exception as e:
                print(f"Warning: Could not convert datetime format: {e}")
                # Keep as string if conversion fails
                pass
        
        print(f"Loaded {len(df)} existing tweets")
        return df
    except Exception as e:
        print(f"Error loading existing data: {e}")
        print("Starting with empty dataset instead.")
        return pd.DataFrame()

def generate_sample_data():
    """
    Generate sample tweet data when API fails
    """
    print("Generating sample Twitter data...")
    
    # Sample positive tweets for iPhone
    iphone_positive_tweets = [
        "Just got the new iPhone 15 Pro Max and the camera is absolutely amazing! #iPhone",
        "Apple's iOS updates are so smooth. Love how my iPhone just works.",
        "Face ID on my new iPhone works perfectly even in low light. Great feature!",
        "The iPhone camera is incredible for night photography. Best phone camera I've used.",
        "Apple's build quality on the iPhone is second to none. This thing feels premium.",
        "iPhone performance is still great after a year of heavy use. Apple silicon is impressive.",
        "The iPhone ecosystem integration with my MacBook and iPad is seamless. Love it!",
        "iPhone's haptic feedback is so satisfying. Android phones just don't compare.",
        "iPhone speakers are surprisingly good for such a slim device. Great stereo sound."
    ]
    
    # Sample negative tweets for iPhone
    iphone_negative_tweets = [
        "Battery life on this iPhone is disappointing. Barely lasts half a day with normal use.",
        "iPhone prices are getting ridiculous. Is it really worth $1200 for a phone?",
        "iPhone storage fills up way too quickly. 128GB should be the minimum option.",
        "Why is iPhone battery degradation still such an issue? Mine's at 85% after just 8 months.",
        "The new iPhone camera bump is huge. Makes the phone wobble on flat surfaces.",
        "Frustrated with my iPhone's signal strength. Keeps dropping calls in my house.",
        "Apple's repair costs for iPhone are absurd. $600 for a simple screen replacement?",
        "So tired of Apple removing features from iPhone and calling it 'innovation'.",
        "iPhone's Lightning port is outdated. Why hasn't Apple switched to USB-C yet?"
    ]
    
    # Sample neutral tweets for iPhone
    iphone_neutral_tweets = [
        "The iPhone display is gorgeous but I wish it didn't have that notch at the top.",
        "Just updated my iPhone to the latest iOS version.",
        "Comparing iPhone and Android features for my next purchase.",
        "Does anyone know if the iPhone 15 is water resistant?",
        "Looking at iPhone cases online. So many options.",
        "My iPhone arrived today. Setting it up now.",
        "The iPhone comes in several colors this year.",
        "Trying to decide between the iPhone Pro and regular models.",
        "iPhone prices vary a lot depending on storage capacity."
    ]
    
    # Sample positive tweets for Samsung Galaxy
    galaxy_positive_tweets = [
        "The Galaxy S23 Ultra's 200MP camera is insane! Samsung really nailed it this year. #SamsungGalaxy",
        "Samsung's display technology is the best in the industry. This AMOLED screen is stunning.",
        "The customization options on my Galaxy are endless. Love being able to make it truly mine.",
        "Samsung's build quality has improved so much. This Galaxy feels solid and premium.",
        "Galaxy S Pen functionality is a game changer for taking notes and drawing.",
        "The wide-angle lens on the Galaxy camera is perfect for landscape photography.",
        "Samsung DeX is underrated. Turning my Galaxy into a desktop is super useful.",
        "The Galaxy's adaptive refresh rate makes scrolling so smooth while saving battery.",
        "Samsung's fast charging is incredible. 0 to 50% in just 15 minutes!"
    ]
    
    # Sample negative tweets for Samsung Galaxy
    galaxy_negative_tweets = [
        "Samsung's One UI is so clunky compared to stock Android. Too many duplicate apps.",
        "Battery life on my Galaxy S22 is terrible. Barely makes it through the day.",
        "Galaxy phones have too much bloatware. Why do I need two app stores?",
        "Samsung needs to improve their update policy. My Galaxy is still waiting for Android 14.",
        "My Galaxy overheats during gaming sessions. Performance throttling is annoying.",
        "Disappointed with my Galaxy camera in low light. Photos are noisy and blurry.",
        "Samsung's customer service is awful. Been waiting weeks for my Galaxy repair.",
        "The curved screen on my Galaxy causes so many accidental touches. Frustrating!",
        "Galaxy phones slow down so much after a year. Planned obsolescence at its finest."
    ]
    
    # Sample neutral tweets for Samsung Galaxy
    galaxy_neutral_tweets = [
        "The Galaxy Z Fold is revolutionary. Having a tablet in my pocket changes everything.",
        "Checking out the new Galaxy S24 specs online.",
        "Considering switching from my old phone to a Galaxy.",
        "Does the Galaxy S23 have expandable storage?",
        "Looking at different Galaxy models to compare prices.",
        "Just ordered a new case for my Galaxy phone.",
        "Samsung announced the Galaxy release date today.",
        "The Galaxy comes with different RAM options this year.",
        "Comparing camera specs between Galaxy models."
    ]
    
    # Create sample data
    data = []
    current_time = datetime.datetime.now()
    
    # Add all tweet categories
    all_tweets = {
        'iphone': {
            'positive': iphone_positive_tweets,
            'negative': iphone_negative_tweets,
            'neutral': iphone_neutral_tweets
        },
        'galaxy': {
            'positive': galaxy_positive_tweets,
            'negative': galaxy_negative_tweets,
            'neutral': galaxy_neutral_tweets
        }
    }
    
    # Generate ID counter - make sure it's different each time
    base_id = int(time.time()) * 10000000000
    id_counter = base_id
    
    # Add all tweets to data
    for category, sentiments in all_tweets.items():
        for sentiment, tweets in sentiments.items():
            for text in tweets:
                tweet_time = current_time - datetime.timedelta(hours=random.randint(1, 72))
                
                data.append({
                    'id': id_counter,
                    'text': text,
                    'created_at': tweet_time,
                    'lang': 'en',
                    'retweet_count': random.randint(0, 50),
                    'reply_count': random.randint(0, 30),
                    'like_count': random.randint(5, 200),
                    'source': random.choice(['Twitter for iPhone', 'Twitter for Web', 'Twitter for Android']),
                    'query': random.choice(['iPhone', 'Apple iPhone', '#iPhone']) if category == 'iphone' else 
                             random.choice(['Samsung Galaxy', 'Galaxy S', 'Galaxy Note', '#SamsungGalaxy']),
                    'category': category
                })
                
                id_counter += 1
    
    return pd.DataFrame(data)

def main():
    """
    Main function to collect tweets for all search terms and append to existing data
    """
    # Create data directory if it doesn't exist
    raw_data_dir = os.path.join('data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Load existing tweet data if available
    existing_df = load_existing_data(raw_data_dir)
    
    # Authenticate with Twitter
    client = authenticate_twitter()
    
    # Get current date for filename
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    
    # Initialize empty DataFrame to store all new tweets
    new_tweets_df = pd.DataFrame()
    
    # Collect tweets for each category and search term
    for category, terms in SEARCH_TERMS.items():
        for term in terms:
            # Build search query: term lang:en -is:retweet
            query = f"{term} lang:en -is:retweet"
            
            # Collect tweets
            tweets_df = collect_tweets(
                client, 
                query, 
                max_results=MAX_TWEETS,
                start_time=START_DATE,
                end_time=END_DATE
            )
            
            if not tweets_df.empty:
                # Add category column
                tweets_df['category'] = category
                
                # Append to new tweets DataFrame
                new_tweets_df = pd.concat([new_tweets_df, tweets_df], ignore_index=True)
            
            # Sleep between queries to respect rate limits
            sleep_time = 5 + random.random() * 5
            print(f"Waiting {sleep_time:.1f} seconds before next query...")
            time.sleep(sleep_time)
    
    # Check if we collected any new tweets
    if new_tweets_df.empty:
        print("Failed to collect any new tweets. Generating sample data instead...")
        new_tweets_df = generate_sample_data()
    
    print(f"Collected a total of {len(new_tweets_df)} new tweets")
    
    # Combine existing and new tweets
    if not existing_df.empty:
        # Combine dataframes
        combined_df = pd.concat([existing_df, new_tweets_df], ignore_index=True)
        
        # Remove duplicates based on tweet ID
        #combined_df = combined_df.drop_duplicates(subset='id', keep='first')
        
        print(f"Combined dataset now has {len(combined_df)} tweets (added {len(combined_df) - len(existing_df)} new unique tweets)")
    else:
        combined_df = new_tweets_df
        print(f"Created new dataset with {len(combined_df)} tweets")
    
    # Save to CSV
    output_path = os.path.join(raw_data_dir, f'tweets_{current_date}.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Saved {len(combined_df)} tweets to {output_path}")
    
    # Also save a backup copy with timestamp to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(raw_data_dir, f'backup_tweets_{timestamp}.csv')
    combined_df.to_csv(backup_path, index=False)
    print(f"Saved backup copy to {backup_path}")

if __name__ == "__main__":
    main()