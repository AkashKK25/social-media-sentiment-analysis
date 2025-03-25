import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_analysis import (
    analyze_sentiment_vader,
    analyze_sentiment_textblob,
    prepare_training_data,
    train_sentiment_model,
    evaluate_sentiment_model,
    save_model
)
import glob
from datetime import datetime
import json

def load_latest_processed_data():
    """
    Load the most recent processed data file
    
    Returns:
        DataFrame with processed tweet data
    """
    # Get list of processed data files
    data_files = glob.glob(os.path.join('data', 'processed', 'processed_tweets_*.csv'))
    
    # If no files found, raise error
    if not data_files:
        raise FileNotFoundError("No processed data files found. Run preprocess_data.py first.")
    
    # Sort by filename (date)
    latest_file = sorted(data_files)[-1]
    print(f"Loading data from {latest_file}")
    
    # Load the data
    df = pd.read_csv(latest_file)
    return df

def analyze_sentiments(df):
    """
    Apply sentiment analysis to tweets
    
    Args:
        df: DataFrame with processed tweet data
        
    Returns:
        DataFrame with sentiment analysis results
    """
    print("Analyzing sentiments...")
    
    # Copy the dataframe to avoid modifying the original
    sentiment_df = df.copy()
    
    # Apply VADER sentiment analysis
    print("Applying VADER sentiment analysis...")
    vader_results = sentiment_df['text'].apply(analyze_sentiment_vader)
    
    # Extract VADER sentiment scores
    sentiment_df['vader_negative'] = vader_results.apply(lambda x: x['neg'])
    sentiment_df['vader_neutral'] = vader_results.apply(lambda x: x['neu'])
    sentiment_df['vader_positive'] = vader_results.apply(lambda x: x['pos'])
    sentiment_df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
    sentiment_df['vader_sentiment'] = vader_results.apply(lambda x: x['sentiment'])
    
    # Apply TextBlob sentiment analysis
    print("Applying TextBlob sentiment analysis...")
    textblob_results = sentiment_df['text'].apply(analyze_sentiment_textblob)
    
    # Extract TextBlob sentiment scores
    sentiment_df['textblob_polarity'] = textblob_results.apply(lambda x: x['polarity'])
    sentiment_df['textblob_subjectivity'] = textblob_results.apply(lambda x: x['subjectivity'])
    sentiment_df['textblob_sentiment'] = textblob_results.apply(lambda x: x['sentiment'])
    
    print("Sentiment analysis complete.")
    return sentiment_df

def train_custom_model(df):
    """
    Train a custom sentiment model
    
    Args:
        df: DataFrame with processed tweet data
        
    Returns:
        Dictionary with model paths and evaluation metrics
    """
    print("Training custom sentiment model...")
    
    try:
        # Prepare training data with a lower minimum sample threshold
        X_train, X_test, y_train, y_test = prepare_training_data(
            df, 
            text_column='cleaned_text',
            min_samples=5  # Reduced from 10 to 5
        )
        
        # Train logistic regression model
        print("Training logistic regression model...")
        logistic_model, logistic_vectorizer = train_sentiment_model(
            X_train, 
            y_train, 
            model_type='logistic'
        )
        
        # Evaluate logistic regression model
        logistic_evaluation = evaluate_sentiment_model(
            logistic_model,
            logistic_vectorizer,
            X_test,
            y_test
        )
        
        print(f"Logistic Regression Accuracy: {logistic_evaluation['accuracy']:.4f}")
        
        # Train SVM model
        print("Training SVM model...")
        svm_model, svm_vectorizer = train_sentiment_model(
            X_train, 
            y_train, 
            model_type='svm'
        )
        
        # Evaluate SVM model
        svm_evaluation = evaluate_sentiment_model(
            svm_model,
            svm_vectorizer,
            X_test,
            y_test
        )
        
        print(f"SVM Accuracy: {svm_evaluation['accuracy']:.4f}")
        
        # Save best model
        if logistic_evaluation['accuracy'] >= svm_evaluation['accuracy']:
            print("Saving logistic regression model...")
            model_path, vectorizer_path = save_model(
                logistic_model,
                logistic_vectorizer,
                output_dir='models'
            )
            best_model_type = 'logistic'
            best_evaluation = logistic_evaluation
        else:
            print("Saving SVM model...")
            model_path, vectorizer_path = save_model(
                svm_model,
                svm_vectorizer,
                output_dir='models'
            )
            best_model_type = 'svm'
            best_evaluation = svm_evaluation
        
        # Return model info
        return {
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'model_type': best_model_type,
            'accuracy': best_evaluation['accuracy'],
            'classification_report': best_evaluation['classification_report']
        }
    
    except ValueError as e:
        print(f"Warning: {e}")
        print("Not enough data for custom model training. Using rule-based sentiment only.")
        
        # Create dummy model info
        os.makedirs('models', exist_ok=True)
        
        # Save a dummy model info file
        model_info = {
            'model_path': None,
            'vectorizer_path': None,
            'model_type': 'rule_based_only',
            'accuracy': None,
            'note': 'Not enough data for custom model training. Using VADER and TextBlob only.'
        }
        
        return model_info

def analyze_sentiment_results(df):
    """
    Analyze and visualize sentiment results
    
    Args:
        df: DataFrame with sentiment analysis results
    """
    print("Analyzing sentiment results...")
    
    # Create output directory
    os.makedirs(os.path.join('data', 'results'), exist_ok=True)
    
    # Save sentiment results
    output_path = os.path.join('data', 'results', f'sentiment_results_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment results to {output_path}")
    
    # Create summary by category
    summary = df.groupby('category').agg({
        'vader_negative': 'mean',
        'vader_neutral': 'mean',
        'vader_positive': 'mean',
        'vader_compound': 'mean',
        'textblob_polarity': 'mean',
        'textblob_subjectivity': 'mean'
    }).reset_index()
    
    # Save summary
    summary_path = os.path.join('data', 'results', f'sentiment_summary_{datetime.now().strftime("%Y%m%d")}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Saved sentiment summary to {summary_path}")
    
    # Count sentiments by category
    sentiment_counts = df.groupby(['category', 'vader_sentiment']).size().reset_index(name='count')
    sentiment_pivot = sentiment_counts.pivot(index='category', columns='vader_sentiment', values='count').fillna(0)
    
    # Calculate percentages
    sentiment_percentages = sentiment_pivot.div(sentiment_pivot.sum(axis=1), axis=0) * 100
    
    # Save sentiment counts
    counts_path = os.path.join('data', 'results', f'sentiment_counts_{datetime.now().strftime("%Y%m%d")}.csv')
    sentiment_counts.to_csv(counts_path, index=False)
    print(f"Saved sentiment counts to {counts_path}")
    
    # Print summary statistics
    print("\nSentiment Summary by Category:")
    print(summary)
    
    print("\nSentiment Distribution by Category:")
    print(sentiment_percentages)
    
    # Generate basic visualizations
    # VADER compound score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='vader_compound', hue='category', bins=30, kde=True)
    plt.title('Distribution of VADER Compound Scores by Category')
    plt.xlabel('Compound Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'results', 'vader_compound_distribution.png'))
    
    # VADER sentiment counts
    plt.figure(figsize=(10, 6))
    sentiment_counts_plot = sentiment_counts.pivot(index='category', columns='vader_sentiment', values='count').fillna(0)
    sentiment_counts_plot.plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'results', 'sentiment_distribution.png'))
    
    # TextBlob polarity vs. subjectivity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='textblob_polarity', y='textblob_subjectivity', hue='category', alpha=0.6)
    plt.title('TextBlob Polarity vs. Subjectivity by Category')
    plt.xlabel('Polarity (Negative to Positive)')
    plt.ylabel('Subjectivity (Objective to Subjective)')
    plt.tight_layout()
    plt.savefig(os.path.join('data', 'results', 'textblob_polarity_subjectivity.png'))
    
    print("Sentiment analysis visualizations saved to data/results directory.")

def main():
    """
    Main function to run sentiment analysis
    """
    # Load processed data
    df = load_latest_processed_data()
    
    # Apply sentiment analysis
    sentiment_df = analyze_sentiments(df)
    
    # Train custom model
    model_info = train_custom_model(sentiment_df)
    
    # Save model info
    model_info_path = os.path.join('models', 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Analyze sentiment results
    analyze_sentiment_results(sentiment_df)
    
    print("Sentiment analysis complete.")

if __name__ == "__main__":
    main()