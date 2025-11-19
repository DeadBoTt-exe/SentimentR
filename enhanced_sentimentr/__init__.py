"""
Enhanced SentimentR - Modern sentiment analysis with Gemini integration
"""
from .core.analyzer import HybridSentimentAnalyzer, SentimentResult
from .core.emotions import EmotionAnalyzer
from .core.aspects import AspectAnalyzer
from .legacy.sentimentr import Sentiment as LegacySentiment

__version__ = "2.0.0"
__author__ = "Yuvraj Singh,By Enhanced Contributors"

# Main exports
__all__ = [
    "HybridSentimentAnalyzer",
    "SentimentResult", 
    "EmotionAnalyzer",
    "AspectAnalyzer",
    "LegacySentiment",
]

# For backward compatibility
Sentiment = LegacySentiment
