# 🚀 Enhanced SentimentR - Quick Start Guide

## ✅ Your System is Ready!

The Enhanced SentimentR system is now fully configured and ready to use. Here are all the ways to get started:

## 🎯 Launch Methods

### 1. 🌐 Web Interface (Streamlit)

**Easiest way (Recommended):**
```bash
cd /home/bhaskar/Capstone/sentimentr
python launch_streamlit.py
```

**Manual way:**
```bash
cd /home/bhaskar/Capstone/sentimentr
export GEMINI_API_KEY="AIzaSyDpjfEYti2BQipYXzygmr5aML_6V5H_0HI"
streamlit run enhanced_sentimentr/web/streamlit_app.py
```

### 2. 💻 Command Line Interface

```bash
cd /home/bhaskar/Capstone/sentimentr
export GEMINI_API_KEY="YOUR GEMINI API KEY"

# Basic analysis
python -m enhanced_sentimentr.cli analyze "I love this product!"

# Full analysis with AI
python -m enhanced_sentimentr.cli analyze "Great food but slow service" --method hybrid --emotions --aspects

# Batch analysis
python -m enhanced_sentimentr.cli batch my_reviews.csv
```

### 3. 🔧 Python Programming

```python
import os
import asyncio

# Set API key
os.environ['GEMINI_API_KEY'] = 'YOUR GEMINI KEY'

# Add to path if needed
import sys
sys.path.insert(0, '/home/bhaskar/Capstone/sentimentr')

from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod

async def analyze_sentiment():
    analyzer = HybridSentimentAnalyzer()
    
    config = AnalysisConfig(
        method=SentimentMethod.HYBRID,
        include_emotions=True,
        include_aspects=True
    )
    
    result = await analyzer.analyze("I love this amazing product!", config)
    
    print(f"Sentiment: {result.polarity:.3f}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Method: {result.method.value}")
    
    if result.emotions:
        print("Emotions:", {e.value: f"{s:.1%}" for e, s in result.emotions.items() if s > 0.1})
    
    if result.aspects:
        print("Aspects:", {a: f"{s:.3f}" for a, s in result.aspects.items()})

# Run the analysis
asyncio.run(analyze_sentiment())
```

### 4. 🌐 REST API Server

```bash
cd /home/bhaskar/Capstone/sentimentr
export GEMINI_API_KEY="YOUR GEMINI KEY"
python -m enhanced_sentimentr.api.main

# Then visit: http://localhost:8000/docs for API documentation
```

## 🧪 Test Everything Works

```bash
cd /home/yuvraj/shitCapstone/sentimentr

# Test API key
export GEMINI_API_KEY="YOUR GEMINI API KEY"
python test_api_key.py

# Run comprehensive demo
python demo.py

# Test installation
python test_installation.py
```

## 🔑 API Key Management

Your API key is saved in: `/home/bhaskar/Capstone/sentimentr/.env`

**For each session, set the environment variable:**
```bash
export GEMINI_API_KEY="YOUR GEMINI API KEY"
```

**Or add to your shell profile permanently:**
```bash
echo 'export GEMINI_API_KEY="YOUR GEMINI API KEY"' >> ~/.bashrc
source ~/.bashrc
```

## 🎮 Quick Examples

### Analyze Social Media Text
```bash
export GEMINI_API_KEY="YOUR GEMINI API KEY"
python -m enhanced_sentimentr.cli analyze "OMG this new phone is absolutely incredible! 😍📱✨" --method hybrid --emotions
```

### Compare Analysis Methods
```bash
export GEMINI_API_KEY="YOUR GEMINI API KEY"

# Rule-based only
python -m enhanced_sentimentr.cli analyze "This product is amazing!" --method rule_based

# AI-powered
python -m enhanced_sentimentr.cli analyze "This product is amazing!" --method gemini

# Best of both
python -m enhanced_sentimentr.cli analyze "This product is amazing!" --method hybrid
```

### Business Review Analysis
```bash
export GEMINI_API_KEY="YOUR GEMINI API KEY"
python -m enhanced_sentimentr.cli analyze "The food quality is excellent and presentation is beautiful, but the service was quite slow and the prices are too high for what you get." --method hybrid --emotions --aspects
```

## 📊 Expected Output

When everything is working correctly, you'll see output like:

```
✅ Positive text: 0.734 (expected: > 0)
🎯 Confidence: 75.2%
⚙️ Method: hybrid
😊 Emotions: Joy (80%), Trust (20%)
🎯 Aspects: Product (positive: 0.950)
```

## 🆘 If Something Goes Wrong

### Import Errors
```bash
# Make sure you're in the right directory
cd /home/bhaskar/Capstone/sentimentr
pwd  # Should show: /home/yuvraj/shitCapstone/sentimentr

# Use the launcher scripts
python launch_streamlit.py  # For web interface
python test_installation.py  # To verify setup
```

### API Key Issues
```bash
# Check if API key is set
echo $GEMINI_API_KEY

# Set it manually
export GEMINI_API_KEY="YOUR GEMINI API KEY"

# Test it works
python test_api_key.py
```

### Permission Issues
```bash
# Make scripts executable
chmod +x launch_streamlit.py
chmod +x setup_api_key.py
chmod +x test_api_key.py
```

## 🎉 You're All Set!

Your Enhanced SentimentR system is now fully operational with:
- ✅ Google Gemini AI integration
- ✅ Multiple analysis methods (rule-based, AI, hybrid)
- ✅ Emotion detection (8 types)
- ✅ Aspect-based sentiment analysis
- ✅ Web interface, CLI, and Python SDK
- ✅ Production-ready deployment options

**Current Status**: 🟢 **FULLY OPERATIONAL**

Enjoy analyzing sentiment at scale! 🎭🚀
