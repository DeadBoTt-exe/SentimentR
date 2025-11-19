"""
Streamlit web interface for Enhanced SentimentR
"""
import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from typing import List, Dict, Any

# Custom imports
import sys
import os

# Add the project root to path to find enhanced_sentimentr
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
    from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod, EmotionType
except ImportError as e:
    st.error(f"""
    ❌ **Import Error**: {e}
    
    **To fix this issue:**
    1. Make sure you're running from the project root directory
    2. Run: `cd /home/yuvraj/shitCapstone/sentimentr`
    3. Then: `streamlit run enhanced_sentimentr/web/streamlit_app.py`
    
    **Or set PYTHONPATH:**
    ```bash
    export PYTHONPATH="/home/yuvraj/shitCapstone/sentimentr:$PYTHONPATH"
    streamlit run enhanced_sentimentr/web/streamlit_app.py
    ```
    """)
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Enhanced SentimentR",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #9b9b9b;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive {
        color: #4caf50;
        font-weight: bold;
    }
    .negative {
        color: #f44336;
        font-weight: bold;
    }
    .neutral {
        color: #ff9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Load and cache the sentiment analyzer"""
    return HybridSentimentAnalyzer()


def get_sentiment_color(polarity: float) -> str:
    """Get color based on sentiment polarity"""
    if polarity > 0.1:
        return "#4caf50"  # Green for positive
    elif polarity < -0.1:
        return "#f44336"  # Red for negative
    else:
        return "#ff9800"  # Orange for neutral


def get_sentiment_emoji(polarity: float) -> str:
    """Get emoji based on sentiment polarity"""
    if polarity > 0.5:
        return "😊"
    elif polarity > 0.1:
        return "🙂"
    elif polarity < -0.5:
        return "😞"
    elif polarity < -0.1:
        return "😐"
    else:
        return "😶"


def create_sentiment_gauge(polarity: float, confidence: float) -> go.Figure:
    """Create a sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=polarity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment Score (Confidence: {confidence:.1%})"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': get_sentiment_color(polarity)},
            'steps': [
                {'range': [-1, -0.1], 'color': "#ffcdd2"},
                {'range': [-0.1, 0.1], 'color': "#fff3e0"},
                {'range': [0.1, 1], 'color': "#c8e6c9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_emotion_chart(emotions: Dict[str, float]) -> go.Figure:
    """Create emotion analysis chart"""
    if not emotions:
        return None
    
    emotion_names = list(emotions.keys())
    emotion_scores = list(emotions.values())
    
    # Filter out emotions with very low scores
    filtered_data = [(name, score) for name, score in zip(emotion_names, emotion_scores) if score > 0.05]
    
    if not filtered_data:
        return None
    
    names, scores = zip(*filtered_data)
    
    fig = px.bar(
        x=list(scores),
        y=list(names),
        orientation='h',
        title="Emotion Analysis",
        labels={'x': 'Intensity', 'y': 'Emotion'},
        color=list(scores),
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_aspect_chart(aspects: Dict[str, float]) -> go.Figure:
    """Create aspect sentiment chart"""
    if not aspects:
        return None
    
    aspect_names = list(aspects.keys())
    aspect_scores = list(aspects.values())
    colors = [get_sentiment_color(score) for score in aspect_scores]
    
    fig = px.bar(
        x=aspect_names,
        y=aspect_scores,
        title="Aspect-Based Sentiment",
        labels={'x': 'Aspect', 'y': 'Sentiment Score'},
        color=aspect_scores,
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    
    fig.update_layout(height=400)
    fig.add_hline(y=0, line_dash="dash", line_color="black", alpha=0.5)
    return fig


async def analyze_text_async(text: str, config: AnalysisConfig, analyzer: HybridSentimentAnalyzer):
    """Async wrapper for text analysis"""
    return await analyzer.analyze(text, config)


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Enhanced SentimentR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Sentiment Analysis with Gemini Integration</p>', unsafe_allow_html=True)
    
    # Load analyzer
    analyzer = load_analyzer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Gemini API Key
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key (Optional)",
        type="password",
        help="Enter your Google Gemini API key for enhanced analysis"
    )
    
    if gemini_api_key:
        analyzer.set_gemini_api_key(gemini_api_key)
        st.sidebar.success("✅ Gemini API key set!")
    
    # Analysis method
    method = st.sidebar.selectbox(
        "Analysis Method",
        options=[method.value for method in SentimentMethod],
        index=2,  # Default to hybrid
        help="Choose the sentiment analysis method"
    )
    
    # Additional options
    include_emotions = st.sidebar.checkbox("Include Emotion Analysis", value=False)
    include_aspects = st.sidebar.checkbox("Include Aspect Analysis", value=False)
    include_explanation = st.sidebar.checkbox("Include Explanation", value=False)
    
    # Create analysis config
    config = AnalysisConfig(
        method=SentimentMethod(method),
        include_emotions=include_emotions,
        include_aspects=include_aspects,
        include_explanation=include_explanation
    )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📋 Batch Analysis", "📊 Analytics"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type your text here...",
            help="Enter any text you want to analyze for sentiment"
        )
        
        # Analysis button
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    try:
                        # Run async analysis
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            analyze_text_async(text_input, config, analyzer)
                        )
                        loop.close()
                        
                        # Display results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Sentiment gauge
                            gauge_fig = create_sentiment_gauge(result.polarity, result.confidence)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            # Basic metrics
                            sentiment_label = (
                                "Positive" if result.polarity > 0.1 
                                else "Negative" if result.polarity < -0.1 
                                else "Neutral"
                            )
                            
                            emoji = get_sentiment_emoji(result.polarity)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{emoji} {sentiment_label}</h3>
                                <p><strong>Polarity:</strong> {result.polarity:.3f}</p>
                                <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
                                <p><strong>Method:</strong> {result.method.value}</p>
                                {f'<p><strong>Processing Time:</strong> {result.processing_time:.3f}s</p>' if result.processing_time else ''}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Additional metrics
                            if result.subjectivity is not None:
                                st.metric("Subjectivity", f"{result.subjectivity:.1%}")
                            
                            if result.intensity is not None:
                                st.metric("Intensity", f"{result.intensity:.1%}")
                            
                            if result.word_count:
                                st.metric("Word Count", result.word_count)
                            
                            if result.sentence_count:
                                st.metric("Sentence Count", result.sentence_count)
                        
                        # Explanation
                        if result.explanation:
                            st.subheader("📋 Analysis Explanation")
                            st.info(result.explanation)
                        
                        # Emotions
                        if result.emotions:
                            st.subheader("😊 Emotion Analysis")
                            emotion_fig = create_emotion_chart(result.emotions)
                            if emotion_fig:
                                st.plotly_chart(emotion_fig, use_container_width=True)
                        
                        # Aspects
                        if result.aspects:
                            st.subheader("🎯 Aspect-Based Sentiment")
                            aspect_fig = create_aspect_chart(result.aspects)
                            if aspect_fig:
                                st.plotly_chart(aspect_fig, use_container_width=True)
                        
                        # Raw data (expandable)
                        with st.expander("🔍 Raw Analysis Data"):
                            st.json(result.to_dict())
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("Batch Analysis")
        
        # Text input methods
        input_method = st.radio(
            "Input Method:",
            ["Text Area", "File Upload"],
            horizontal=True
        )
        
        texts = []
        
        if input_method == "Text Area":
            batch_text = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="Text 1\nText 2\nText 3\n...",
                help="Enter each text on a separate line"
            )
            
            if batch_text.strip():
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Upload a text file or CSV",
                type=['txt', 'csv'],
                help="Upload a .txt file (one text per line) or .csv file with a 'text' column"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            texts = df['text'].astype(str).tolist()
                        else:
                            st.error("CSV file must have a 'text' column")
                    else:
                        content = uploaded_file.read().decode('utf-8')
                        texts = [line.strip() for line in content.split('\n') if line.strip()]
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Show preview
        if texts:
            st.write(f"Found {len(texts)} texts to analyze")
            with st.expander("Preview texts"):
                for i, text in enumerate(texts[:5]):  # Show first 5
                    st.write(f"{i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
                if len(texts) > 5:
                    st.write(f"... and {len(texts) - 5} more")
        
        # Batch analysis
        if st.button("🔄 Analyze Batch", type="primary") and texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                try:
                    # Run batch analysis
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    batch_result = loop.run_until_complete(
                        analyzer.batch_analyze(texts, config)
                    )
                    loop.close()
                    
                    # Display batch results
                    st.success(f"✅ Analysis complete! Processed {batch_result.successful_analyses} texts successfully.")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Texts", batch_result.total_texts)
                    with col2:
                        st.metric("Successful", batch_result.successful_analyses)
                    with col3:
                        st.metric("Failed", batch_result.failed_analyses)
                    with col4:
                        st.metric("Avg Time", f"{batch_result.average_processing_time:.3f}s")
                    
                    # Results visualization
                    if batch_result.results:
                        polarities = [r.polarity for r in batch_result.results]
                        confidences = [r.confidence for r in batch_result.results]
                        
                        # Distribution charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_hist = px.histogram(
                                x=polarities,
                                title="Sentiment Distribution",
                                labels={'x': 'Polarity', 'y': 'Count'},
                                nbins=20
                            )
                            fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            fig_scatter = px.scatter(
                                x=polarities,
                                y=confidences,
                                title="Polarity vs Confidence",
                                labels={'x': 'Polarity', 'y': 'Confidence'},
                                color=polarities,
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Results table
                        results_data = []
                        for i, (text, result) in enumerate(zip(texts, batch_result.results)):
                            results_data.append({
                                'Index': i + 1,
                                'Text': text[:100] + '...' if len(text) > 100 else text,
                                'Sentiment': 'Positive' if result.polarity > 0.1 else 'Negative' if result.polarity < -0.1 else 'Neutral',
                                'Polarity': round(result.polarity, 3),
                                'Confidence': round(result.confidence, 3),
                                'Method': result.method.value
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.subheader("📊 Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"sentiment_analysis_results_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Batch analysis failed: {str(e)}")
    
    with tab3:
        st.header("Analytics & Statistics")
        
        # System status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 System Status")
            status_data = {
                "Gemini Available": "✅ Yes" if analyzer.is_gemini_available() else "❌ No",
                "Default Method": config.method.value,
                "Emotions Enabled": "✅ Yes" if config.include_emotions else "❌ No",
                "Aspects Enabled": "✅ Yes" if config.include_aspects else "❌ No",
            }
            
            for key, value in status_data.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.subheader("📈 Cache Statistics")
            try:
                cache_stats = analyzer.get_cache_stats()
                if cache_stats:
                    st.json(cache_stats)
                else:
                    st.write("No cache statistics available")
            except Exception as e:
                st.write(f"Could not retrieve cache stats: {str(e)}")
        
        # Clear cache button
        if st.button("🗑️ Clear All Caches"):
            try:
                analyzer.clear_caches()
                st.success("✅ Caches cleared successfully!")
            except Exception as e:
                st.error(f"Failed to clear caches: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Enhanced SentimentR v2.0.0 | 
            <a href='https://github.com/Yuvraj-ai' target='_blank'>GitHub</a> | 
            Built with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
