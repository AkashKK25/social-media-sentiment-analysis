import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
#import pyLDAvis.sklearn
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import matplotlib.pyplot as plt
import os

def extract_ngrams(texts, n_gram_range=(1, 1), min_df=2, max_df=0.95, max_features=None):
    """
    Extract most common n-grams from texts
    
    Args:
        texts: List of text documents
        n_gram_range: Range of n-grams to extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        max_features: Maximum number of features to extract
        
    Returns:
        DataFrame with n-grams and their frequencies
    """
    # Adjust min_df based on corpus size
    corpus_size = len(texts)
    
    # For very small datasets, use lower min_df
    if corpus_size < 20:
        adaptive_min_df = 1
    elif corpus_size < 50:
        adaptive_min_df = 2
    else:
        adaptive_min_df = min_df
        
    print(f"Using min_df={adaptive_min_df} for corpus size {corpus_size}")
    
    # Initialize vectorizer
    count_vec = CountVectorizer(
        ngram_range=n_gram_range,
        min_df=adaptive_min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words='english'
    )
    
    try:
        # Fit and transform texts
        ngram_counts = count_vec.fit_transform(texts)
        
        # Get feature names
        feature_names = count_vec.get_feature_names_out()
        
        # Sum counts across documents
        ngram_sums = ngram_counts.sum(axis=0)
        
        # Convert to array
        ngram_sums = np.asarray(ngram_sums).flatten()
        
        # Create DataFrame
        ngram_df = pd.DataFrame({
            'ngram': feature_names,
            'count': ngram_sums
        })
        
        # Sort by count
        ngram_df = ngram_df.sort_values('count', ascending=False).reset_index(drop=True)
        
        return ngram_df
    
    except ValueError as e:
        print(f"Warning: {e}")
        print("Creating empty DataFrame instead.")
        return pd.DataFrame(columns=['ngram', 'count'])
    
def prepare_texts_for_gensim(texts):
    """
    Prepare texts for Gensim LDA modeling
    
    Args:
        texts: List of preprocessed text documents
        
    Returns:
        corpus, dictionary, and preprocessed texts
    """
    # Split texts into lists of words
    tokenized_texts = [text.split() for text in texts if isinstance(text, str) and len(text.strip()) > 0]
    
    # Filter out empty texts
    tokenized_texts = [tokens for tokens in tokenized_texts if len(tokens) > 1]
    
    # Create dictionary
    dictionary = corpora.Dictionary(tokenized_texts)
    
    # For small datasets, use less restrictive filtering
    if len(tokenized_texts) < 20:
        dictionary.filter_extremes(no_below=1, no_above=0.9)
    else:
        dictionary.filter_extremes(no_below=2, no_above=0.8)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    return corpus, dictionary, tokenized_texts

def run_lda_gensim(corpus, dictionary, tokenized_texts, num_topics=5, random_state=42):
    """
    Run LDA topic modeling using Gensim
    
    Args:
        corpus: Document-term matrix
        dictionary: Gensim dictionary
        tokenized_texts: List of tokenized texts
        num_topics: Number of topics to extract
        random_state: Random state for reproducibility
        
    Returns:
        LDA model, topics, and coherence score
    """
    # Train LDA model
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=random_state,
        workers=1  # Set to number of CPU cores - 1
    )
    
    # Extract topics
    topics = lda_model.print_topics(num_words=10)
    
    # Calculate coherence score
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    
    return lda_model, topics, coherence_score

def find_optimal_topics(corpus, dictionary, tokenized_texts, start=2, end=12, step=1):
    """
    Find optimal number of topics by coherence score
    
    Args:
        corpus: Document-term matrix
        dictionary: Gensim dictionary
        tokenized_texts: List of tokenized texts
        start: Starting number of topics
        end: Ending number of topics
        step: Step size
        
    Returns:
        List of coherence scores and optimal number of topics
    """
    coherence_scores = []
    
    for num_topics in range(start, end + 1, step):
        print(f"Testing {num_topics} topics...")
        
        # Train LDA model
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            random_state=42,
            workers=1
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Store score
        coherence_scores.append((num_topics, coherence_score))
        print(f"Coherence score: {coherence_score}")
    
    # Find optimal number of topics
    optimal_topics = max(coherence_scores, key=lambda x: x[1])[0]
    
    return coherence_scores, optimal_topics

def visualize_lda_gensim(lda_model, corpus, dictionary, output_path):
    """
    Create interactive LDA visualization using pyLDAvis
    
    Args:
        lda_model: Trained LDA model
        corpus: Document-term matrix
        dictionary: Gensim dictionary
        output_path: Path to save visualization HTML
    """
    # Prepare visualization
    vis_data = pyLDAvis.gensim_models.prepare(
        lda_model, 
        corpus, 
        dictionary,
        sort_topics=False
    )
    
    # Save visualization
    pyLDAvis.save_html(vis_data, output_path)
    print(f"LDA visualization saved to {output_path}")

def extract_document_topics(lda_model, corpus, threshold=0.1):
    """
    Extract dominant topics for each document
    
    Args:
        lda_model: Trained LDA model
        corpus: Document-term matrix
        threshold: Minimum probability threshold
        
    Returns:
        List of (document_index, topic_id, probability) tuples
    """
    # Get topic distribution for each document
    document_topics = []
    
    for i, doc_topics in enumerate(lda_model[corpus]):
        # Sort by probability
        doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        
        # Filter by threshold
        doc_topics = [(topic_id, prob) for topic_id, prob in doc_topics if prob >= threshold]
        
        # Add document index
        doc_topics = [(i, topic_id, prob) for topic_id, prob in doc_topics]
        
        document_topics.extend(doc_topics)
    
    return document_topics

def extract_top_terms_per_topic(lda_model, dictionary, num_terms=20):
    """
    Extract top terms for each topic
    
    Args:
        lda_model: Trained LDA model
        dictionary: Gensim dictionary
        num_terms: Number of terms to extract per topic
        
    Returns:
        Dictionary of {topic_id: [(term, probability), ...]}
    """
    # Initialize dictionary
    topic_terms = {}
    
    # Extract top terms for each topic
    for topic_id in range(lda_model.num_topics):
        # Get topic terms with probabilities
        topic = lda_model.get_topic_terms(topic_id, topn=num_terms)
        
        # Convert term IDs to actual terms
        terms = [(dictionary[term_id], prob) for term_id, prob in topic]
        
        # Store in dictionary
        topic_terms[topic_id] = terms
    
    return topic_terms

def extract_key_phrases(texts, nlp=None):
    """
    Extract key phrases from texts using spaCy
    
    Args:
        texts: List of text documents
        nlp: spaCy NLP model (optional)
        
    Returns:
        List of key phrases for each document
    """
    # Load spaCy model if not provided
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    
    # Initialize list to store phrases
    all_phrases = []
    
    # Define POS patterns for key phrases
    patterns = [
        ['ADJ', 'NOUN'],
        ['NOUN', 'NOUN'],
        ['ADJ', 'ADJ', 'NOUN'],
        ['ADV', 'ADJ', 'NOUN'],
        ['VERB', 'NOUN'],
        ['NOUN', 'VERB', 'NOUN']
    ]
    
    # Process each text
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            all_phrases.append([])
            continue
        
        # Parse text
        doc = nlp(text)
        
        # Extract phrases
        phrases = []
        
        for sentence in doc.sents:
            # Get tokens with their POS tags
            tokens = [(token.text.lower(), token.pos_) for token in sentence 
                      if token.text.lower() not in STOP_WORDS and token.is_alpha]
            
            # Look for patterns
            for i in range(len(tokens) - 1):
                # Check for adjective-noun pattern
                if i < len(tokens) - 1 and tokens[i][1] == 'ADJ' and tokens[i+1][1] == 'NOUN':
                    phrases.append(f"{tokens[i][0]} {tokens[i+1][0]}")
                
                # Check for noun-noun pattern
                if i < len(tokens) - 1 and tokens[i][1] == 'NOUN' and tokens[i+1][1] == 'NOUN':
                    phrases.append(f"{tokens[i][0]} {tokens[i+1][0]}")
                
                # Check for longer patterns
                if i < len(tokens) - 2:
                    # ADJ-ADJ-NOUN
                    if tokens[i][1] == 'ADJ' and tokens[i+1][1] == 'ADJ' and tokens[i+2][1] == 'NOUN':
                        phrases.append(f"{tokens[i][0]} {tokens[i+1][0]} {tokens[i+2][0]}")
                    
                    # ADV-ADJ-NOUN
                    if tokens[i][1] == 'ADV' and tokens[i+1][1] == 'ADJ' and tokens[i+2][1] == 'NOUN':
                        phrases.append(f"{tokens[i][0]} {tokens[i+1][0]} {tokens[i+2][0]}")
                    
                    # VERB-NOUN
                    if tokens[i][1] == 'VERB' and tokens[i+1][1] == 'NOUN':
                        phrases.append(f"{tokens[i][0]} {tokens[i+1][0]}")
        
        # Add phrases for this document
        all_phrases.append(phrases)
    
    return all_phrases

def analyze_topic_sentiment(df, topic_column, sentiment_column):
    """
    Analyze sentiment by topic
    
    Args:
        df: DataFrame with topics and sentiment
        topic_column: Column name with topic assignments
        sentiment_column: Column name with sentiment labels
        
    Returns:
        DataFrame with sentiment distribution by topic
    """
    # Count sentiment by topic
    topic_sentiment = df.groupby([topic_column, sentiment_column]).size().reset_index(name='count')
    
    # Pivot to get sentiment distribution
    topic_sentiment_pivot = topic_sentiment.pivot(
        index=topic_column,
        columns=sentiment_column,
        values='count'
    ).fillna(0)
    
    # Calculate percentages
    topic_sentiment_pct = topic_sentiment_pivot.div(topic_sentiment_pivot.sum(axis=1), axis=0) * 100
    
    return topic_sentiment_pivot, topic_sentiment_pct