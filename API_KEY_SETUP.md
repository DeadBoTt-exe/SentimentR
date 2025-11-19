# 🔑 Enhanced SentimentR API Key Configuration Guide

Enhanced SentimentR supports multiple ways to provide your Google Gemini API key for AI-powered sentiment analysis.

## 🚀 Quick Setup

### Method 1: Environment Variable (Recommended)

Set the environment variable in your terminal:

```bash
# Linux/Mac
export GEMINI_API_KEY="your-actual-api-key-here"

# Windows Command Prompt
set GEMINI_API_KEY=your-actual-api-key-here

# Windows PowerShell
$env:GEMINI_API_KEY="your-actual-api-key-here"
```

Then use the system normally:
```bash
python -m enhanced_sentimentr.cli analyze "I love this product!" --method hybrid
```

### Method 2: Python Code

Pass the API key directly when creating the analyzer:

```python
from enhanced_sentimentr import HybridSentimentAnalyzer

# Initialize with API key
analyzer = HybridSentimentAnalyzer(gemini_api_key="your-actual-api-key-here")

# Use the analyzer
result = await analyzer.analyze("I love this product!")
print(f"Sentiment: {result.polarity:.3f}")
```

### Method 3: Configuration File

Create a `.env` file in your project directory:

```bash
# Create .env file
echo "GEMINI_API_KEY=your-actual-api-key-here" > .env
```

### Method 4: CLI with Environment File

You can also use environment files with the CLI:

```bash
# Create environment file
echo "GEMINI_API_KEY=your-actual-api-key-here" > gemini.env

# Source it before running
source gemini.env
python -m enhanced_sentimentr.cli analyze "Great product!" --method hybrid
```

## 🔍 How to Get a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key
5. Use it with Enhanced SentimentR

## 🧪 Testing Your API Key

Run this simple test to verify your API key works:

```python
import asyncio
from enhanced_sentimentr import HybridSentimentAnalyzer, AnalysisConfig, SentimentMethod

async def test_gemini():
    # Replace with your actual API key
    analyzer = HybridSentimentAnalyzer(gemini_api_key="your-api-key-here")
    
    config = AnalysisConfig(method=SentimentMethod.GEMINI)
    result = await analyzer.analyze("This is amazing!", config)
    
    print(f"✅ Gemini API is working!")
    print(f"Sentiment: {result.polarity:.3f}")
    print(f"Method: {result.method.value}")

# Run the test
asyncio.run(test_gemini())
```

## 🎯 Different Usage Scenarios

### Scenario 1: Development Environment

For development, use environment variables:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export GEMINI_API_KEY="your-api-key-here"

# Restart terminal or source the file
source ~/.bashrc

# Now all commands will use Gemini
python -m enhanced_sentimentr.cli analyze "Test text" --method hybrid
```

### Scenario 2: Production Deployment

For Docker/production, pass as environment variable:

```bash
# Docker run
docker run -e GEMINI_API_KEY="your-api-key" enhanced-sentimentr

# Docker compose (add to environment section)
environment:
  - GEMINI_API_KEY=your-api-key-here
```

### Scenario 3: Jupyter Notebook

For notebooks, set the environment variable or pass directly:

```python
import os
import asyncio
from enhanced_sentimentr import HybridSentimentAnalyzer

# Option 1: Set environment variable in notebook
os.environ['GEMINI_API_KEY'] = 'your-api-key-here'
analyzer = HybridSentimentAnalyzer()

# Option 2: Pass directly
analyzer = HybridSentimentAnalyzer(gemini_api_key='your-api-key-here')

# Use the analyzer
result = await analyzer.analyze("This notebook is great!")
print(f"Sentiment: {result.polarity:.3f}")
```

### Scenario 4: Web Application

For the Streamlit web app, you have several options:

**Option 1: Use the launcher script (Recommended)**
```bash
# This handles all path setup automatically
python launch_streamlit.py
```

**Option 2: Set environment and run from project root**
```bash
# Make sure you're in the project root directory
cd /home/yuvraj/shitCapstone/sentimentr

# Set API key
export GEMINI_API_KEY="your-api-key-here"

# Launch Streamlit
streamlit run enhanced_sentimentr/web/streamlit_app.py
```

**Option 3: Set PYTHONPATH**
```bash
export PYTHONPATH="/home/yuvraj/shitCapstone/sentimentr:$PYTHONPATH"
export GEMINI_API_KEY="your-api-key-here"
streamlit run enhanced_sentimentr/web/streamlit_app.py
```

## 🛡️ Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables in production**
3. **Rotate your API keys regularly**
4. **Set up API key restrictions in Google Cloud Console**
5. **Monitor your API usage and costs**

## 🔧 Troubleshooting

### Issue: "No Gemini API key provided"
**Solution**: Set the `GEMINI_API_KEY` environment variable or pass it to the constructor

### Issue: "Gemini analysis failed"
**Solutions**:
- Check your API key is valid
- Verify you have credits/quota remaining
- Check your internet connection
- The system will automatically fall back to rule-based analysis

### Issue: API key not being detected
**Solutions**:
- Restart your terminal after setting environment variables
- Check spelling: it must be exactly `GEMINI_API_KEY`
- Use `echo $GEMINI_API_KEY` to verify it's set

### Issue: "ModuleNotFoundError: No module named 'enhanced_sentimentr'" (Streamlit)
**Solutions**:
- Use the launcher script: `python launch_streamlit.py`
- Make sure you're in the project root: `cd /home/yuvraj/shitCapstone/sentimentr`
- Set PYTHONPATH: `export PYTHONPATH="/path/to/sentimentr:$PYTHONPATH"`
- Check the current directory with `pwd` - should be the sentimentr project root

## 🎮 Quick Test Commands

Test different methods with your API key:

```bash
# Set your API key first
export GEMINI_API_KEY="your-actual-api-key-here"

# Test rule-based (no API key needed)
python -m enhanced_sentimentr.cli analyze "I love this!" --method rule_based

# Test Gemini (requires API key)
python -m enhanced_sentimentr.cli analyze "I love this!" --method gemini

# Test hybrid (best of both)
python -m enhanced_sentimentr.cli analyze "I love this!" --method hybrid

# Test with all features
python -m enhanced_sentimentr.cli analyze "Great product but slow delivery" --method hybrid --emotions --aspects
```

## 🎯 Without API Key

Enhanced SentimentR works perfectly without an API key using the rule-based analyzer:

```bash
# Rule-based analysis (no API key needed)
python -m enhanced_sentimentr.cli analyze "I love this product!" --method rule_based --emotions --aspects
```

The system will automatically fall back to rule-based analysis if no API key is provided, so you can use most features without any setup!
