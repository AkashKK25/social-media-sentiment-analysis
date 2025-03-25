import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import json
import os
from collections import Counter

def generate_wordcloud(text, width=800, height=400, max_words=100, background_color='white'):
    """
    Generate a WordCloud from text
    
    Args:
        text: Text to generate wordcloud from
        width: Width of wordcloud
        height: Height of wordcloud
        max_words: Maximum number of words to include
        background_color: Background color
        
    Returns:
        WordCloud object
    """
    return WordCloud(
        width=width, 
        height=height, 
        background_color=background_color,
        max_words=max_words,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)

def get_top_entities(df, entity_type='PRODUCT', n=10):
    """
    Get top entities of a specific type
    
    Args:
        df: DataFrame with entities column
        entity_type: Type of entity to extract
        n: Number of top entities to return
        
    Returns:
        DataFrame with top entities and counts
    """
    # Extract entities
    all_entities = []
    
    for entities_str in df['entities'].dropna():
        try:
            entities = eval(entities_str)
            if entity_type in entities:
                all_entities.extend(entities[entity_type])
        except:
            continue
    
    # Count entities
    entity_counts = Counter(all_entities)
    
    # Convert to DataFrame
    entity_df = pd.DataFrame(entity_counts.most_common(n), columns=['entity', 'count'])
    
    return entity_df

def get_sentiment_colors():
    """
    Get color mapping for sentiment labels
    
    Returns:
        Dictionary mapping sentiment labels to colors
    """
    return {
        'positive': '#2ECC71',  # Green
        'neutral': '#3498DB',   # Blue
        'negative': '#E74C3C'   # Red
    }

def create_sentiment_distribution_chart(df, category_col='category', sentiment_col='vader_sentiment'):
    """
    Create sentiment distribution chart
    
    Args:
        df: DataFrame with sentiment data
        category_col: Column name for categories
        sentiment_col: Column name for sentiment
        
    Returns:
        Plotly figure
    """
    # Count sentiments by category
    sentiment_counts = df.groupby([category_col, sentiment_col]).size().reset_index(name='count')
    
    # Create bar chart
    fig = px.bar(
        sentiment_counts,
        x=category_col,
        y='count',
        color=sentiment_col,
        barmode='group',
        color_discrete_map=get_sentiment_colors(),
        labels={'count': 'Number of Tweets', category_col: 'Category', sentiment_col: 'Sentiment'}
    )
    
    return fig

def create_sentiment_time_chart(df, date_col='date', category_col='category', sentiment_col='vader_sentiment'):
    """
    Create sentiment over time chart
    
    Args:
        df: DataFrame with sentiment data
        date_col: Column name for date
        category_col: Column name for categories
        sentiment_col: Column name for sentiment
        
    Returns:
        Plotly figure
    """
    # Group by date, category, and sentiment
    time_sentiment = df.groupby([pd.Grouper(key=date_col, freq='D'), category_col, sentiment_col]).size().reset_index(name='count')
    
    # Create line chart
    fig = px.line(
        time_sentiment,
        x=date_col,
        y='count',
        color=category_col,
        line_dash=sentiment_col,
        facet_row=sentiment_col,
        labels={'count': 'Number of Tweets', date_col: 'Date', category_col: 'Category'},
        height=600
    )
    
    fig.update_layout(hovermode="x unified")
    
    return fig

def create_feature_chart(df, feature_cols, category_col='category'):
    """
    Create feature mentions chart
    
    Args:
        df: DataFrame with feature mention columns
        feature_cols: List of feature column names
        category_col: Column name for categories
        
    Returns:
        Plotly figure or None
    """
    # Prepare data for visualization
    feature_data = []
    
    for category in df[category_col].unique():
        category_df = df[df[category_col] == category]
        
        for feature in feature_cols:
            feature_name = feature.replace('mentions_', '')
            total_mentions = category_df[feature].sum()
            
            if total_mentions > 0:
                feature_data.append({
                    'category': category,
                    'feature': feature_name.capitalize(),
                    'mentions': total_mentions
                })
    
    feature_df = pd.DataFrame(feature_data)
    
    if feature_df.empty:
        return None
    
    # Create bar chart
    fig = px.bar(
        feature_df,
        x='feature',
        y='mentions',
        color='category',
        barmode='group',
        labels={'mentions': 'Number of Mentions', 'feature': 'Feature', 'category': 'Category'}
    )
    
    return fig

def create_topic_term_chart(terms, max_terms=15):
    """
    Create chart for topic terms
    
    Args:
        terms: List of (term, probability) tuples
        max_terms: Maximum number of terms to include
        
    Returns:
        Plotly figure
    """
    # Create term data for visualization
    term_data = pd.DataFrame(
        [(t['term'], t['prob']) for t in terms[:max_terms]],
        columns=['term', 'probability']
    )
    
    # Create horizontal bar chart
    fig = px.bar(
        term_data,
        y='term',
        x='probability',
        orientation='h',
        labels={'probability': 'Probability', 'term': 'Term'},
        height=400
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig

def create_engagement_chart(df, engagement_cols, category_col='category', sentiment_col='vader_sentiment'):
    """
    Create engagement metrics chart
    
    Args:
        df: DataFrame with engagement metrics
        engagement_cols: List of engagement column names
        category_col: Column name for categories
        sentiment_col: Column name for sentiment
        
    Returns:
        Plotly figure
    """
    # Create subplot with multiple metrics
    from plotly.subplots import make_subplots
    
    # Generate titles
    titles = [col.replace('_count', '').capitalize() for col in engagement_cols]
    
    # Create subplot
    fig = make_subplots(
        rows=1,
        cols=len(engagement_cols),
        subplot_titles=titles,
        shared_yaxes=True
    )
    
    # Add traces for each metric
    for i, metric in enumerate(engagement_cols):
        for category in df[category_col].unique():
            category_data = df[df[category_col] == category]
            
            for sentiment in category_data[sentiment_col].unique():
                sentiment_data = category_data[category_data[sentiment_col] == sentiment]
                
                fig.add_trace(
                    go.Box(
                        y=sentiment_data[metric],
                        name=f"{category}-{sentiment}",
                        legendgroup=f"{category}-{sentiment}",
                        showlegend=(i == 0),
                        boxmean=True
                    ),
                    row=1,
                    col=i+1
                )
    
    fig.update_layout(
        height=500,
        boxmode='group',
        legend_title_text="Category-Sentiment"
    )
    
    # Log scale for better visualization
    fig.update_yaxes(type="log")
    
    return fig