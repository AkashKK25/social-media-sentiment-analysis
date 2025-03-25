import os
import sys

# Get the absolute path of the current file
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
project_dir = os.path.dirname(dashboard_dir)
# Add the project directory to the Python path
sys.path.append(project_dir)

# Now update your data paths to be relative to the project directory
data_path = os.path.join(project_dir, 'data', 'results')
model_path = os.path.join(project_dir, 'models')


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import json
import glob
import os
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set page config
st.set_page_config(
    page_title="Social Media Sentiment Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
data_path = os.path.join('..', 'data', 'results')
model_path = os.path.join('..', 'models')

# Load data
@st.cache_data
def load_data():
    # Find latest sentiment results
    sentiment_files = glob.glob(os.path.join(data_path, 'sentiment_results_*.csv'))
    if not sentiment_files:
        st.error("No sentiment data found. Please run the sentiment analysis script first.")
        return None, None, None, None
    
    latest_sentiment_file = sorted(sentiment_files)[-1]
    df = pd.read_csv(latest_sentiment_file)
    
    # Convert date columns to datetime
    for col in ['created_at', 'date', 'processed_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Load feature data if available
    feature_file = os.path.join(data_path, 'tweets_with_features.csv')
    if os.path.exists(feature_file):
        df_features = pd.read_csv(feature_file)
        
        # Merge with main dataframe if needed
        if 'id' in df_features.columns and 'id' in df.columns:
            df = pd.merge(df, df_features[['id', 'key_phrases', 'total_feature_mentions'] + 
                                        [col for col in df_features.columns if col.startswith('mentions_')]],
                         on='id', how='left')
    
    # Load topic data
    topic_summary_file = os.path.join(data_path, 'topic_summary.json')
    if os.path.exists(topic_summary_file):
        with open(topic_summary_file, 'r') as f:
            topic_summary = json.load(f)
    else:
        topic_summary = {}
    
    # Load topic terms
    topic_terms = {}
    for category in df['category'].unique():
        topic_terms_file = os.path.join(data_path, f'{category}_top_terms.json')
        if os.path.exists(topic_terms_file):
            with open(topic_terms_file, 'r') as f:
                topic_terms[category] = json.load(f)
    
    # Load category-specific dataframes with topics
    category_dfs = {}
    for category in df['category'].unique():
        category_file = os.path.join(data_path, f'{category}_with_topics.csv')
        if os.path.exists(category_file):
            category_dfs[category] = pd.read_csv(category_file)
    
    return df, topic_summary, topic_terms, category_dfs

# Load data
df, topic_summary, topic_terms, category_dfs = load_data()

# Title and description
st.title("📱 Smartphone Sentiment Analysis Dashboard")
st.markdown("""
This dashboard analyzes Twitter sentiment about iPhones and Samsung Galaxy smartphones.
Explore sentiment trends, popular topics, and key features mentioned by users.
""")

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Date range selector
if df is not None and 'date' in df.columns:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
    else:
        filtered_df = df
else:
    filtered_df = df

# Category selector
if filtered_df is not None and 'category' in filtered_df.columns:
    categories = filtered_df['category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=categories,
        default=categories
    )
    
    if selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

# Sentiment selector
if filtered_df is not None and 'vader_sentiment' in filtered_df.columns:
    sentiments = filtered_df['vader_sentiment'].unique()
    selected_sentiments = st.sidebar.multiselect(
        "Select Sentiments",
        options=sentiments,
        default=sentiments
    )
    
    if selected_sentiments:
        filtered_df = filtered_df[filtered_df['vader_sentiment'].isin(selected_sentiments)]

st.sidebar.header("Author")
st.sidebar.subheader("Akash Kumar Kondaparthi")

# Main dashboard
if filtered_df is not None and not filtered_df.empty:
    # Overview metrics
    st.header("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tweets", len(filtered_df))
    
    with col2:
        positive_pct = (filtered_df['vader_sentiment'] == 'positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
    
    with col3:
        negative_pct = (filtered_df['vader_sentiment'] == 'negative').mean() * 100
        st.metric("Negative Sentiment", f"{negative_pct:.1f}%")
    
    with col4:
        avg_engagement = filtered_df['like_count'].mean()
        st.metric("Avg. Likes", f"{avg_engagement:.1f}")
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution by Category")
    
    sentiment_counts = filtered_df.groupby(['category', 'vader_sentiment']).size().reset_index(name='count')
    
    fig = px.bar(
        sentiment_counts,
        x='category',
        y='count',
        color='vader_sentiment',
        barmode='group',
        color_discrete_map={
            'positive': '#2ECC71',
            'neutral': '#3498DB',
            'negative': '#E74C3C'
        },
        labels={'count': 'Number of Tweets', 'category': 'Category', 'vader_sentiment': 'Sentiment'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Sentiment over time
    st.subheader("Sentiment Trends Over Time")
    
    time_sentiment = filtered_df.groupby([pd.Grouper(key='date', freq='D'), 'category', 'vader_sentiment']).size().reset_index(name='count')
    
    fig = px.line(
        time_sentiment,
        x='date',
        y='count',
        color='category',
        line_dash='vader_sentiment',
        facet_row='vader_sentiment',
        labels={'count': 'Number of Tweets', 'date': 'Date', 'category': 'Category'},
        height=600
    )
    
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # Features and Topics
    st.header("📋 Features and Topics")
    
    # Two columns layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Common Features Mentioned")
        
        if 'mentions_display' in filtered_df.columns:
            feature_cols = [col for col in filtered_df.columns if col.startswith('mentions_')]
            
            if feature_cols:
                # Prepare data for visualization
                feature_data = []
                
                for category in filtered_df['category'].unique():
                    category_df = filtered_df[filtered_df['category'] == category]
                    
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
                
                if not feature_df.empty:
                    fig = px.bar(
                        feature_df,
                        x='feature',
                        y='mentions',
                        color='category',
                        barmode='group',
                        labels={'mentions': 'Number of Mentions', 'feature': 'Feature', 'category': 'Category'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Word Cloud")
        
        # Word cloud for selected categories
        if selected_categories:
            selected_category = st.selectbox("Select Category for Word Cloud", options=selected_categories)
            
            category_df = filtered_df[filtered_df['category'] == selected_category]
            
            if not category_df.empty and 'cleaned_text' in category_df.columns:
                # Combine all cleaned text
                text = ' '.join(category_df['cleaned_text'].dropna())
                
                if text:
                    # Generate wordcloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='viridis',
                        contour_width=1,
                        contour_color='steelblue'
                    ).generate(text)
                    
                    # Display the wordcloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
    
    # Topic analysis
    if topic_summary and selected_categories:
        st.header("📚 Topic Analysis")
        
        for category in selected_categories:
            if category in topic_summary:
                st.subheader(f"Topics for {category}")
                
                # Display topic info
                num_topics = topic_summary[category]['optimal_topics']
                st.write(f"Number of topics: {num_topics}")
                
                # Display topics and top terms
                if category in topic_terms:
                    # Create tabs for each topic
                    topic_tabs = st.tabs([f"Topic {i+1}" for i in range(num_topics)])
                    
                    for i, tab in enumerate(topic_tabs):
                        with tab:
                            topic_id = str(i)
                            
                            if topic_id in topic_terms[category]:
                                # Display top terms
                                terms = topic_terms[category][topic_id]
                                
                                # Create term data for visualization
                                term_data = pd.DataFrame(
                                    [(t['term'], t['prob']) for t in terms[:15]],
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
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display example tweets for this topic
                                if category in category_dfs:
                                    cat_df = category_dfs[category]
                                    topic_tweets = cat_df[cat_df['dominant_topic'] == int(topic_id)]
                                    
                                    if not topic_tweets.empty:
                                        # Sort by topic probability
                                        top_tweets = topic_tweets.sort_values('topic_probability', ascending=False).head(3)
                                        
                                        st.write("Example tweets for this topic:")
                                        for _, tweet in top_tweets.iterrows():
                                            st.markdown(f"- _{tweet['text']}_")
                                            
                                # Topic sentiment analysis if available
                                if 'topic_sentiment' in topic_terms[category]:
                                    st.write("Sentiment distribution for this topic:")
                                    # (Add sentiment visualization here)
    
    # Engagement analysis
    st.header("👥 Engagement Analysis")
    
    # Engagement by sentiment
    st.subheader("Engagement by Sentiment")
    
    engagement_metrics = ['like_count', 'retweet_count', 'reply_count']
    
    # Create subplot with three metrics
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=("Likes", "Retweets", "Replies"),
        shared_yaxes=True
    )
    
    for i, metric in enumerate(engagement_metrics):
        for category in filtered_df['category'].unique():
            category_data = filtered_df[filtered_df['category'] == category]
            
            for sentiment in category_data['vader_sentiment'].unique():
                sentiment_data = category_data[category_data['vader_sentiment'] == sentiment]
                
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
    
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tweet explorer
    st.header("🔍 Tweet Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_likes = st.number_input("Minimum Likes", value=0, min_value=0)
    
    with col2:
        sentiment_filter = st.selectbox(
            "Filter by Sentiment",
            options=['All'] + list(filtered_df['vader_sentiment'].unique())
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            options=['Likes', 'Retweets', 'Replies', 'Date (Newest)', 'Date (Oldest)']
        )
    
    # Apply filters
    explorer_df = filtered_df[filtered_df['like_count'] >= min_likes]
    
    if sentiment_filter != 'All':
        explorer_df = explorer_df[explorer_df['vader_sentiment'] == sentiment_filter]
    
    # Apply sorting
    if sort_by == 'Likes':
        explorer_df = explorer_df.sort_values('like_count', ascending=False)
    elif sort_by == 'Retweets':
        explorer_df = explorer_df.sort_values('retweet_count', ascending=False)
    elif sort_by == 'Replies':
        explorer_df = explorer_df.sort_values('reply_count', ascending=False)
    elif sort_by == 'Date (Newest)':
        explorer_df = explorer_df.sort_values('created_at', ascending=False)
    elif sort_by == 'Date (Oldest)':
        explorer_df = explorer_df.sort_values('created_at', ascending=True)
    
    # Display tweets
    for i, row in explorer_df.head(10).iterrows():
        with st.expander(f"{row['text'][:100]}..."):
            st.write(f"**Full text:** {row['text']}")
            st.write(f"**Category:** {row['category']}")
            st.write(f"**Sentiment:** {row['vader_sentiment']} (VADER Score: {row['vader_compound']:.3f})")
            st.write(f"**Posted on:** {row['created_at']}")
            st.write(f"**Likes:** {row['like_count']} | **Retweets:** {row['retweet_count']} | **Replies:** {row['reply_count']}")
            
            # Display key phrases if available
            if 'key_phrases' in row and row['key_phrases']:
                try:
                    phrases = eval(row['key_phrases'])
                    if phrases:
                        st.write("**Key phrases:**")
                        st.write(', '.join(phrases))
                except:
                    pass
    
    # Download data option
    st.subheader("Download Data")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(filtered_df)
    
    st.download_button(
        "Download Filtered Data as CSV",
        csv,
        "smartphone_sentiment_data.csv",
        "text/csv",
        key='download-csv'
    )

else:
    st.error("No data to display based on the selected filters. Please adjust your filters or run the analysis scripts.")

# Footer
st.markdown("---")
st.markdown("""
**Social Media Sentiment Analysis Dashboard** | Created by Akash Kumar Kondaparthi
""")