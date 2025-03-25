import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER
    
    Args:
        text: String of text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    scores = analyzer.polarity_scores(text)
    
    # Add sentiment label
    if scores['compound'] >= 0.05:
        scores['sentiment'] = 'positive'
    elif scores['compound'] <= -0.05:
        scores['sentiment'] = 'negative'
    else:
        scores['sentiment'] = 'neutral'
    
    return scores

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob
    
    Args:
        text: String of text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Create TextBlob object
    blob = TextBlob(text)
    
    # Get polarity and subjectivity
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Add sentiment label
    if polarity > 0.1:
        sentiment = 'positive'
    elif polarity < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment': sentiment
    }

def prepare_training_data(df, text_column, min_samples=5):  # Changed from 10 to 5
    """
    Prepare training data for machine learning model
    
    Args:
        df: DataFrame with preprocessed tweets
        text_column: Column name containing preprocessed text
        min_samples: Minimum number of samples per category
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Use VADER sentiment as labels
    df['vader_sentiment'] = df['text'].apply(
        lambda x: analyze_sentiment_vader(x)['sentiment']
    )
    
    # Count samples per label
    label_counts = df['vader_sentiment'].value_counts()
    print(f"Sentiment distribution: {dict(label_counts)}")
    
    # Check if we have enough samples
    if min(label_counts) < min_samples:
        raise ValueError(f"Not enough samples for training. Min count: {min(label_counts)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], 
        df['vader_sentiment'],
        test_size=0.2,
        random_state=42,
        stratify=df['vader_sentiment']
    )
    
    return X_train, X_test, y_train, y_test


def train_sentiment_model(X_train, y_train, model_type='logistic'):
    """
    Train a sentiment analysis model
    
    Args:
        X_train: Training features (text)
        y_train: Training labels
        model_type: Type of model to train ('logistic' or 'svm')
        
    Returns:
        Trained model and vectorizer
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    # Transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'svm':
        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train_tfidf, y_train)
    
    return model, vectorizer

def evaluate_sentiment_model(model, vectorizer, X_test, y_test):
    """
    Evaluate sentiment analysis model
    
    Args:
        model: Trained model
        vectorizer: TF-IDF vectorizer
        X_test: Test features (text)
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'y_pred': y_pred
    }

def save_model(model, vectorizer, output_dir='../models'):
    """
    Save trained model and vectorizer
    
    Args:
        model: Trained model
        vectorizer: TF-IDF vectorizer
        output_dir: Directory to save model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    import time
    timestamp = int(time.time())
    
    # Save model
    model_path = os.path.join(output_dir, f'sentiment_model_{timestamp}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    vectorizer_path = os.path.join(output_dir, f'vectorizer_{timestamp}.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    
    return model_path, vectorizer_path

def load_model(model_path, vectorizer_path):
    """
    Load trained model and vectorizer
    
    Args:
        model_path: Path to model file
        vectorizer_path: Path to vectorizer file
        
    Returns:
        Loaded model and vectorizer
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment of text
    
    Args:
        text: String of text
        model: Trained model
        vectorizer: TF-IDF vectorizer
        
    Returns:
        Predicted sentiment
    """
    # Transform text
    text_tfidf = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    return prediction

def compare_sentiment_methods(text):
    """
    Compare sentiment analysis methods on a single text
    
    Args:
        text: String of text
        
    Returns:
        Dictionary with sentiment results from different methods
    """
    # VADER sentiment
    vader_sentiment = analyze_sentiment_vader(text)
    
    # TextBlob sentiment
    textblob_sentiment = analyze_sentiment_textblob(text)
    
    return {
        'text': text,
        'vader': vader_sentiment,
        'textblob': textblob_sentiment
    }