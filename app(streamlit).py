
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
from io import BytesIO
import os
from datetime import datetime

# Import custom modules
from src.cleaner import InstagramCommentCleaner
from src.sentiment import SentimentAnalyzer
from src.clustering import AudienceClusterer
from src.insights import InsightGenerator

# Page configuration
st.set_page_config(
    page_title="Instagram Reels Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Title and description
st.title("📊 Instagram Reels Sentiment Analyzer")
st.markdown("""
    Analyze audience sentiment and behavior from Instagram Reel comments.
    Upload your scraped comments CSV or use the demo data to get started.
""")

# Sidebar for data input
with st.sidebar:
    st.header("📥 Data Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload CSV (Recommended)", "🎲 Use Demo Data", "🔗 Apify Integration (Demo)"]
    )
    
    if input_method == "📁 Upload CSV (Recommended)":
        uploaded_file = st.file_uploader(
            "Upload Instagram comments CSV",
            type=['csv'],
            help="CSV should have a column named 'text' containing comments"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✅ Loaded {len(df)} comments")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    elif input_method == "🎲 Use Demo Data":
        # Create demo data
        demo_comments = pd.DataFrame({
            'text': [
                "This reel is amazing! Love the editing 🔥",
                "Best content ever! Keep it up",
                "Not really my style but okay",
                "Terrible audio quality, can't hear anything",
                "Great tips! Very helpful",
                "Why does everyone like this? It's boring",
                "This made my day! 😍",
                "Meh, seen better content",
                "Absolutely fantastic work!",
                "The music is too loud, can't focus",
                "Finally some good content on this topic",
                "This is exactly what I needed, thank you!",
                "Not impressed, overhyped",
                "Love the energy! More please",
                "The editing is top notch"
            ] * 10  # Repeat to have more data
        })
        st.session_state.data = demo_comments
        st.success("✅ Loaded demo data")
    
    else:  # Apify Integration
        st.info("""
            **Apify Integration Demo**
            
            To use live scraping:
            1. Go to apify.com
            2. Use Instagram Comment Scraper
            3. Download CSV and upload above
            
            *Note: Live scraping disabled for stability*
        """)

# Main content area
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Data preview
    with st.expander("📄 Data Preview", expanded=False):
        st.dataframe(df.head(10))
        st.write(f"Total comments: {len(df)}")
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Run Complete Analysis", 
            type="primary", 
            use_container_width=True
        )
    
    if analyze_button:
        with st.spinner("Running analysis pipeline... This may take a moment."):
            # Initialize components
            cleaner = InstagramCommentCleaner()
            sentiment_analyzer = SentimentAnalyzer()
            clusterer = AudienceClusterer(n_clusters=4)
            insight_gen = InsightGenerator()
            
            # Step 1: Clean data
            progress_bar = st.progress(0)
            st.write("🧹 Cleaning comments...")
            df_clean = cleaner.clean_dataframe(df)
            progress_bar.progress(25)
            
            # Step 2: Sentiment analysis
            st.write("🎭 Analyzing sentiment...")
            df_analyzed = sentiment_analyzer.add_sentiment_to_df(df_clean)
            progress_bar.progress(50)
            
            # Step 3: Clustering
            st.write("👥 Clustering audiences...")
            df_clustered = clusterer.fit_clusters(df_analyzed)
            cluster_keywords = clusterer.get_cluster_keywords(df_clustered)
            cluster_labels = clusterer.label_clusters(df_clustered, cluster_keywords)
            progress_bar.progress(75)
            
            # Step 4: Generate insights
            st.write("💡 Generating insights...")
            sentiment_percentages = insight_gen.calculate_sentiment_percentages(df_clustered)
            summary = insight_gen.generate_summary(df_clustered, cluster_labels)
            cluster_descriptions = insight_gen.generate_cluster_descriptions(cluster_labels)
            
            # Store in session state
            st.session_state.analyzed_data = df_clustered
            st.session_state.sentiment_percentages = sentiment_percentages
            st.session_state.cluster_labels = cluster_labels
            st.session_state.cluster_descriptions = cluster_descriptions
            st.session_state.summary = summary
            st.session_state.cluster_keywords = cluster_keywords
            st.session_state.analyzed = True
            
            progress_bar.progress(100)
            st.success("✅ Analysis complete!")
    
    # Display results if analyzed
    if st.session_state.get('analyzed', False):
        df_analyzed = st.session_state.analyzed_data
        
        # Summary section
        st.header("📈 Key Insights")
        st.info(st.session_state.summary)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Sentiment Analysis", 
            "👥 Audience Clusters", 
            "☁️ Word Clouds",
            "📋 Detailed Data"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                sentiment_counts = df_analyzed['sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'neutral': '#95a5a6',
                        'negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart with confidence
                avg_confidence = df_analyzed.groupby('sentiment')['confidence'].mean().reset_index()
                fig_bar = px.bar(
                    avg_confidence,
                    x='sentiment',
                    y='confidence',
                    title="Average Confidence by Sentiment",
                    color='sentiment',
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'neutral': '#95a5a6',
                        'negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Sentiment scores distribution
            st.subheader("Sentiment Score Distribution")
            fig_hist = go.Figure()
            for sentiment in ['positive', 'neutral', 'negative']:
                fig_hist.add_trace(go.Histogram(
                    x=df_analyzed[f'{sentiment}_score'],
                    name=sentiment.capitalize(),
                    opacity=0.7
                ))
            fig_hist.update_layout(barmode='overlay', title="Distribution of Sentiment Scores")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            st.subheader("Audience Segments")
            
            # Display cluster cards
            cols = st.columns(len(st.session_state.cluster_descriptions))
            for idx, (cluster_id, desc) in enumerate(st.session_state.cluster_descriptions.items()):
                with cols[idx]:
                    with st.container(border=True):
                        st.markdown(desc)
            
            # Cluster visualization
            st.subheader("Cluster Visualization")
            fig_clusters = clusterer.visualize_clusters(df_analyzed)
            st.pyplot(fig_clusters)
            
            # Cluster keywords
            st.subheader("Top Keywords by Cluster")
            keywords_df = pd.DataFrame([
                {
                    'Cluster': f"Cluster {cid}",
                    'Label': st.session_state.cluster_labels[cid]['label'],
                    'Top Keywords': ', '.join(keywords[:5])
                }
                for cid, keywords in st.session_state.cluster_keywords.items()
            ])
            st.dataframe(keywords_df, use_container_width=True)
        
        with tab3:
            st.subheader("Word Clouds by Sentiment")
            
            col1, col2, col3 = st.columns(3)
            
            for col, sentiment in zip([col1, col2, col3], ['positive', 'neutral', 'negative']):
                with col:
                    st.write(f"**{sentiment.capitalize()} Comments**")
                    
                    # Get texts for this sentiment
                    texts = ' '.join(df_analyzed[df_analyzed['sentiment'] == sentiment]['cleaned_text'].tolist())
                    
                    if texts.strip():
                        wordcloud = WordCloud(
                            width=400, height=300,
                            background_color='white',
                            colormap='viridis',
                            max_words=50
                        ).generate(texts)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        st.write("No comments in this category")
        
        with tab4:
            st.subheader("Analyzed Comments")
            
            # Display analyzed data
            display_cols = ['cleaned_text', 'sentiment', 'confidence', 'cluster', 
                          'length', 'word_count', 'emoji_count']
            available_cols = [col for col in display_cols if col in df_analyzed.columns]
            
            st.dataframe(df_analyzed[available_cols], use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Download CSV
                csv = df_analyzed.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">📥 Download Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                # Download summary report (simplified)
                if st.button("📄 Generate PDF Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Instagram Sentiment Analysis Report", ln=1, align='C')
                    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
                    pdf.cell(200, 10, txt="", ln=1)
                    
                    # Add summary
                    pdf.multi_cell(0, 10, txt=st.session_state.summary)
                    
                    # Save PDF
                    pdf_output = BytesIO()
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="analysis_report.pdf">📥 Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

else:
    # Welcome message when no data loaded
    st.info("👈 Please upload a CSV file or load demo data from the sidebar to begin analysis.")
    
    # Show sample of expected format
    st.markdown("### Expected CSV Format")
    sample_df = pd.DataFrame({
        'text': [
            'This is a sample comment',
            'Another comment example',
            'Love this content! 🔥'
        ]
    })
    st.dataframe(sample_df)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>📱 Instagram Reels Sentiment Analyzer | Built with Streamlit & Hugging Face</p>
        <p style='color: #666; font-size: 0.8em;'>⚠️ For educational purposes only. Respect Instagram's terms of service.</p>
    </div>
""", unsafe_allow_html=True)
