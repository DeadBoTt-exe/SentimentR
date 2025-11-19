"""
Simple wrapper for the original sentiment analyzer
"""

# Import the original sentiment functionality
import sys
import os

# Add the project root to sys.path to access sentimentr package
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

try:
    from sentimentr.sentimentr import Sentiment
    
    def get_sentiment_score(text, subjectivity=False, verbose=False):
        """Wrapper function for the original sentiment analyzer"""
        return Sentiment.get_polarity_score(text, subjectivity=subjectivity, verbose=verbose)
        
except ImportError as e:
    # Fallback if import fails
    def get_sentiment_score(text, subjectivity=False, verbose=False):
        """Fallback sentiment analyzer"""
        # Simple word-based sentiment
        positive_words = ['good', 'great', 'awesome', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst']
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        if pos_count > neg_count:
            score = min(0.8, pos_count * 0.3)
        elif neg_count > pos_count:
            score = max(-0.8, neg_count * -0.3)
        else:
            score = 0.0
            
        if subjectivity:
            return {
                'polarity': score,
                'pos portion': pos_count / max(len(words), 1),
                'neg portion': neg_count / max(len(words), 1),
                'neutral portion': max(0, (len(words) - pos_count - neg_count) / max(len(words), 1))
            }
        return score
