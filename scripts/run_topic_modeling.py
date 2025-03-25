import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from datetime import datetime
import spacy
from topic_modeling import (
    extract_ngrams,
    prepare_texts_for_gensim,
    run_lda_gensim,
    find_optimal_topics,
    visualize_lda_gensim,
    extract_document_topics,
    extract_top_terms_per_topic,
    extract_key_phrases,
    analyze_topic_sentiment
)
import pickle
import json

def load_sentiment_results():
    """
    Load the most recent sentiment results
    
    Returns:
        DataFrame with sentiment analysis results
    """
    # Get list of sentiment result files
    data_files = glob.glob(os.path.join('data', 'results', 'sentiment_results_*.csv'))
    
    # If no files found, raise error
    if not data_files:
        raise FileNotFoundError("No sentiment results found. Run run_sentiment_analysis.py first.")
    
    # Sort by filename (date)
    latest_file = sorted(data_files)[-1]
    print(f"Loading sentiment results from {latest_file}")
    
    # Load the data
    df = pd.read_csv(latest_file)
    return df

def extract_common_phrases(df):
    """
    Extract common phrases from tweets
    
    Args:
        df: DataFrame with tweet data
        
    Returns:
        Dictionary with common phrases by category
    """
    print("Extracting common phrases...")
    
    # Dictionary to store results
    phrase_results = {}
    
    # Extract unigrams and bigrams for each category
    for category in df['category'].unique():
        # Filter data by category
        category_df = df[df['category'] == category]
        
        print(f"\nProcessing category: {category} with {len(category_df)} tweets")
        
        # Skip if too few documents
        if len(category_df) < 5:
            print(f"Warning: Not enough documents for category '{category}'. Skipping.")
            phrase_results[category] = {
                'unigrams': pd.DataFrame(columns=['ngram', 'count']),
                'bigrams': pd.DataFrame(columns=['ngram', 'count']),
                'trigrams': pd.DataFrame(columns=['ngram', 'count'])
            }
            continue
        
        try:
            # Extract unigrams
            unigrams = extract_ngrams(
                category_df['cleaned_text'].dropna(),
                n_gram_range=(1, 1),
                min_df=1,  # Lower min_df to 1
                max_df=0.95,
                max_features=100
            )
            
            # Extract bigrams
            bigrams = extract_ngrams(
                category_df['cleaned_text'].dropna(),
                n_gram_range=(2, 2),
                min_df=1,  # Lower min_df to 1
                max_df=0.95,
                max_features=100
            )
            
            # Extract trigrams
            trigrams = extract_ngrams(
                category_df['cleaned_text'].dropna(),
                n_gram_range=(3, 3),
                min_df=1,  # Lower min_df to 1
                max_df=0.95,
                max_features=50
            )
            
            # Store results
            phrase_results[category] = {
                'unigrams': unigrams,
                'bigrams': bigrams,
                'trigrams': trigrams
            }
            
            # Print top 5 bigrams for category
            if not bigrams.empty:
                print(f"\nTop 5 bigrams for {category}:")
                print(bigrams.head(5))
            else:
                print(f"\nNo bigrams found for {category}")
                
        except Exception as e:
            print(f"Error processing category '{category}': {e}")
            # Create empty DataFrames for this category
            phrase_results[category] = {
                'unigrams': pd.DataFrame(columns=['ngram', 'count']),
                'bigrams': pd.DataFrame(columns=['ngram', 'count']),
                'trigrams': pd.DataFrame(columns=['ngram', 'count'])
            }
    
    return phrase_results

def run_topic_modeling(df):
    """
    Run topic modeling on tweets
    
    Args:
        df: DataFrame with tweet data
        
    Returns:
        Dictionary with topic modeling results
    """
    print("Running topic modeling...")
    
    # Dictionary to store results
    topic_results = {}
    
    # Create output directory for visualizations
    os.makedirs(os.path.join('data', 'results', 'topic_vis'), exist_ok=True)
    
    # Run topic modeling for each category
    for category in df['category'].unique():
        print(f"\nRunning topic modeling for {category}...")
        
        try:
            # Filter data by category
            category_df = df[df['category'] == category]
            
            # Skip if too few documents
            if len(category_df) < 10:
                print(f"Warning: Not enough documents for category '{category}'. Using simplified topic modeling.")
                
                # Create simplified results
                topic_results[category] = {
                    'lda_model': None,
                    'dictionary': None,
                    'corpus': None,
                    'optimal_topics': 2,
                    'coherence_score': 0.0,
                    'topics': [(0, "default topic 0"), (1, "default topic 1")],
                    'top_terms': {0: [], 1: []},
                    'document_topics': [],
                    'topic_sentiment': pd.DataFrame(),
                    'topic_sentiment_pct': pd.DataFrame(),
                    'df_with_topics': category_df.copy()
                }
                continue
            
            # Prepare texts for gensim
            texts = category_df['cleaned_text'].dropna().tolist()
            texts = [text for text in texts if isinstance(text, str) and len(text.split()) > 3]
            
            if len(texts) < 5:
                print(f"Warning: Not enough valid documents for category '{category}'. Using simplified topic modeling.")
                
                # Create simplified results
                topic_results[category] = {
                    'lda_model': None,
                    'dictionary': None,
                    'corpus': None,
                    'optimal_topics': 2,
                    'coherence_score': 0.0,
                    'topics': [(0, "default topic 0"), (1, "default topic 1")],
                    'top_terms': {0: [], 1: []},
                    'document_topics': [],
                    'topic_sentiment': pd.DataFrame(),
                    'topic_sentiment_pct': pd.DataFrame(),
                    'df_with_topics': category_df.copy()
                }
                continue
            
            corpus, dictionary, tokenized_texts = prepare_texts_for_gensim(texts)
            
            # Use a fixed number of topics for small datasets
            if len(texts) < 20:
                optimal_topics = 2
                print(f"Small dataset detected. Using fixed number of topics: {optimal_topics}")
            else:
                # Find optimal number of topics
                try:
                    coherence_scores, optimal_topics = find_optimal_topics(
                        corpus, 
                        dictionary, 
                        tokenized_texts,
                        start=2,
                        end=min(5, len(texts) // 2),  # Limit max topics based on corpus size
                        step=1
                    )
                except Exception as e:
                    print(f"Error finding optimal topics: {e}")
                    optimal_topics = 2
            
            print(f"Optimal number of topics for {category}: {optimal_topics}")
            
            # Run LDA with optimal number of topics
            try:
                lda_model, topics, coherence_score = run_lda_gensim(
                    corpus, 
                    dictionary, 
                    tokenized_texts,
                    num_topics=optimal_topics
                )
                
                print(f"Coherence score: {coherence_score}")
                print("Topics:")
                for topic_id, topic in topics:
                    print(f"Topic {topic_id}: {topic}")
                
                # Create visualization
                vis_path = os.path.join('data', 'results', 'topic_vis', f'lda_vis_{category}.html')
                visualize_lda_gensim(lda_model, corpus, dictionary, vis_path)
                
                # Extract document topics
                document_topics = extract_document_topics(lda_model, corpus, threshold=0.2)
                
                # Extract top terms per topic
                top_terms = extract_top_terms_per_topic(lda_model, dictionary, num_terms=20)
                
                # Add dominant topic to dataframe
                if document_topics:
                    # Create mapping from document index to (topic_id, probability)
                    doc_topic_map = {}
                    for doc_idx, topic_id, prob in document_topics:
                        if doc_idx not in doc_topic_map or prob > doc_topic_map[doc_idx][1]:
                            doc_topic_map[doc_idx] = (topic_id, prob)
                    
                    # Add to dataframe
                    category_df_with_topics = category_df.copy()
                    category_df_with_topics = category_df_with_topics.reset_index(drop=True)
                    category_df_with_topics['dominant_topic'] = category_df_with_topics.index.map(
                        lambda x: doc_topic_map.get(x, (-1, 0))[0] if x < len(doc_topic_map) else -1
                    )
                    category_df_with_topics['topic_probability'] = category_df_with_topics.index.map(
                        lambda x: doc_topic_map.get(x, (-1, 0))[1] if x < len(doc_topic_map) else 0
                    )
                    
                    # Analyze sentiment by topic
                    try:
                        topic_sentiment, topic_sentiment_pct = analyze_topic_sentiment(
                            category_df_with_topics,
                            'dominant_topic',
                            'vader_sentiment'
                        )
                        
                        print("\nSentiment by topic:")
                        print(topic_sentiment_pct)
                    except Exception as e:
                        print(f"Error analyzing sentiment by topic: {e}")
                        topic_sentiment = pd.DataFrame()
                        topic_sentiment_pct = pd.DataFrame()
                else:
                    category_df_with_topics = category_df.copy()
                    category_df_with_topics['dominant_topic'] = -1
                    category_df_with_topics['topic_probability'] = 0
                    topic_sentiment = pd.DataFrame()
                    topic_sentiment_pct = pd.DataFrame()
                
                # Store results
                topic_results[category] = {
                    'lda_model': lda_model,
                    'dictionary': dictionary,
                    'corpus': corpus,
                    'optimal_topics': optimal_topics,
                    'coherence_score': coherence_score,
                    'topics': topics,
                    'top_terms': top_terms,
                    'document_topics': document_topics,
                    'topic_sentiment': topic_sentiment,
                    'topic_sentiment_pct': topic_sentiment_pct,
                    'df_with_topics': category_df_with_topics
                }
                
            except Exception as e:
                print(f"Error running LDA for category '{category}': {e}")
                
                # Create empty results for this category
                topic_results[category] = {
                    'lda_model': None,
                    'dictionary': None,
                    'corpus': None,
                    'optimal_topics': 2,
                    'coherence_score': 0.0,
                    'topics': [(0, "default topic 0"), (1, "default topic 1")],
                    'top_terms': {0: [], 1: []},
                    'document_topics': [],
                    'topic_sentiment': pd.DataFrame(),
                    'topic_sentiment_pct': pd.DataFrame(),
                    'df_with_topics': category_df.copy()
                }
        
        except Exception as e:
            print(f"Unexpected error for category '{category}': {e}")
            
            # Create empty results for this category
            topic_results[category] = {
                'lda_model': None,
                'dictionary': None,
                'corpus': None,
                'optimal_topics': 2,
                'coherence_score': 0.0,
                'topics': [(0, "default topic 0"), (1, "default topic 1")],
                'top_terms': {0: [], 1: []},
                'document_topics': [],
                'topic_sentiment': pd.DataFrame(),
                'topic_sentiment_pct': pd.DataFrame(),
                'df_with_topics': pd.DataFrame() if category not in df['category'].unique() else df[df['category'] == category].copy()
            }
    
    return topic_results

def extract_features(df):
    """
    Extract key features mentioned in tweets
    
    Args:
        df: DataFrame with tweet data
        
    Returns:
        DataFrame with extracted features
    """
    print("Extracting key features...")
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Extract key phrases
    df['key_phrases'] = extract_key_phrases(df['text'].dropna(), nlp)
    
    # Count mentions of specific features
    feature_categories = {
        'display': ['screen', 'display', 'resolution', 'oled', 'lcd', 'retina', 'amoled'],
        'camera': ['camera', 'lens', 'photo', 'picture', 'video', 'megapixel', 'pixel', 'portrait'],
        'battery': ['battery', 'charge', 'charging', 'power', 'life', 'fast charging'],
        'performance': ['speed', 'fast', 'slow', 'lag', 'performance', 'processor', 'chip'],
        'design': ['design', 'look', 'feel', 'build', 'quality', 'premium', 'glass', 'metal'],
        'software': ['ios', 'android', 'update', 'software', 'os', 'interface'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'afford', 'value']
    }
    
    # Function to count feature mentions
    def count_features(text, features):
        if not isinstance(text, str):
            return 0
        text = text.lower()
        return sum(1 for feature in features if feature in text)
    
    # Count mentions for each feature category
    for category, features in feature_categories.items():
        df[f'mentions_{category}'] = df['text'].apply(lambda x: count_features(x, features))
    
    # Total feature mentions
    df['total_feature_mentions'] = df[[f'mentions_{category}' for category in feature_categories]].sum(axis=1)
    
    return df

def save_results(phrase_results, topic_results, df_with_features):
    """
    Save analysis results
    
    Args:
        phrase_results: Dictionary with phrase extraction results
        topic_results: Dictionary with topic modeling results
        df_with_features: DataFrame with extracted features
    """
    print("Saving results...")
    
    # Create output directory
    os.makedirs(os.path.join('data', 'results'), exist_ok=True)
    
    # Save phrase results
    for category, results in phrase_results.items():
        for ngram_type, ngram_df in results.items():
            output_path = os.path.join('data', 'results', f'{category}_{ngram_type}.csv')
            ngram_df.to_csv(output_path, index=False)
    
    # Save topic results
    topic_summary = {}
    for category, results in topic_results.items():
        # Save topics
        topics_dict = {}
        for topic_id, topic_str in results['topics']:
            topics_dict[int(topic_id)] = topic_str
        
        topic_summary_path = os.path.join('data', 'results', f'{category}_topics.json')
        with open(topic_summary_path, 'w') as f:
            json.dump(topics_dict, f, indent=4)
        
        # Save top terms
        top_terms_dict = {}
        for topic_id, terms in results['top_terms'].items():
            top_terms_dict[int(topic_id)] = [{'term': term, 'prob': float(prob)} for term, prob in terms]
        
        top_terms_path = os.path.join('data', 'results', f'{category}_top_terms.json')
        with open(top_terms_path, 'w') as f:
            json.dump(top_terms_dict, f, indent=4)
        
        # Save topic sentiment
        if not results['topic_sentiment'].empty:
            topic_sentiment_path = os.path.join('data', 'results', f'{category}_topic_sentiment.csv')
            results['topic_sentiment'].to_csv(topic_sentiment_path)
        
        # Save dataframe with topics
        df_with_topics_path = os.path.join('data', 'results', f'{category}_with_topics.csv')
        results['df_with_topics'].to_csv(df_with_topics_path, index=False)
        
        # Add summary info
        topic_summary[category] = {
            'optimal_topics': results['optimal_topics'],
            'coherence_score': float(results['coherence_score']),
            'topic_count': len(topics_dict)
        }
    
    # Save topic summary
    topic_summary_path = os.path.join('data', 'results', 'topic_summary.json')
    with open(topic_summary_path, 'w') as f:
        json.dump(topic_summary, f, indent=4)
    
    # Save dataframe with features
    features_path = os.path.join('data', 'results', 'tweets_with_features.csv')
    df_with_features.to_csv(features_path, index=False)
    
    # Save LDA models
    os.makedirs(os.path.join('models', 'topic'), exist_ok=True)
    for category, results in topic_results.items():
        # Save LDA model
        model_path = os.path.join('models', 'topic', f'lda_model_{category}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(results['lda_model'], f)
        
        # Save dictionary
        dict_path = os.path.join('models', 'topic', f'dictionary_{category}.pkl')
        with open(dict_path, 'wb') as f:
            pickle.dump(results['dictionary'], f)
    
    print("Results saved.")

def main():
    """
    Main function to run topic modeling and feature extraction
    """
    # Load sentiment results
    df = load_sentiment_results()
    
    # Extract common phrases
    phrase_results = extract_common_phrases(df)
    
    # Run topic modeling
    topic_results = run_topic_modeling(df)
    
    # Extract features
    df_with_features = extract_features(df)
    
    # Save results
    save_results(phrase_results, topic_results, df_with_features)
    
    print("Topic modeling and feature extraction complete.")

if __name__ == "__main__":
    main()