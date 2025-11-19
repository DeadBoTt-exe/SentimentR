"""
Aspect-based sentiment analysis module
"""
import time
import logging
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .models import AnalysisConfig
from .sentiment_wrapper import get_sentiment_score


logger = logging.getLogger(__name__)


class AspectAnalyzer:
    """
    Performs aspect-based sentiment analysis to identify sentiment towards specific aspects/entities
    """
    
    def __init__(self):
        """Initialize aspect analyzer"""
        self.common_aspects = self._load_common_aspects()
        self.aspect_patterns = self._load_aspect_patterns()
    
    def _load_common_aspects(self) -> Dict[str, List[str]]:
        """Load common aspect categories and their keywords"""
        return {
            "service": [
                "service", "staff", "employee", "waiter", "waitress", "server", 
                "customer service", "support", "help", "assistance", "team"
            ],
            "quality": [
                "quality", "build", "construction", "material", "durability", 
                "craftsmanship", "finish", "design", "performance"
            ],
            "price": [
                "price", "cost", "expensive", "cheap", "affordable", "value", 
                "money", "budget", "pricing", "fee", "charge", "rate"
            ],
            "food": [
                "food", "meal", "dish", "taste", "flavor", "delicious", "cuisine", 
                "recipe", "ingredient", "cooking", "chef", "menu", "appetizer", 
                "main course", "dessert", "breakfast", "lunch", "dinner"
            ],
            "location": [
                "location", "place", "area", "neighborhood", "address", "venue", 
                "spot", "setting", "environment", "atmosphere", "ambiance"
            ],
            "product": [
                "product", "item", "goods", "merchandise", "device", "gadget", 
                "tool", "equipment", "machine", "software", "app", "application"
            ],
            "delivery": [
                "delivery", "shipping", "transport", "arrival", "time", "speed", 
                "fast", "slow", "quick", "delayed", "prompt", "timely"
            ],
            "user_interface": [
                "interface", "ui", "ux", "design", "layout", "navigation", 
                "menu", "button", "screen", "display", "usability", "ease of use"
            ],
            "features": [
                "feature", "functionality", "capability", "option", "setting", 
                "tool", "function", "ability", "characteristic", "property"
            ],
            "size": [
                "size", "big", "small", "large", "huge", "tiny", "compact", 
                "spacious", "roomy", "cramped", "dimensions", "length", "width"
            ]
        }
    
    def _load_aspect_patterns(self) -> List[str]:
        """Load regex patterns for aspect extraction"""
        return [
            r'\b(?:the|this|that)\s+(\w+)\s+(?:is|was|are|were)\s+(\w+)',
            r'\b(\w+)\s+(?:is|was|are|were)\s+(?:very|really|quite|extremely)?\s*(\w+)',
            r'\blove\s+(?:the\s+)?(\w+)',
            r'\bhate\s+(?:the\s+)?(\w+)',
            r'\b(\w+)\s+(?:quality|performance|service)',
            r'\b(?:good|bad|great|terrible|awful|amazing|excellent)\s+(\w+)',
            r'\b(\w+)\s+(?:sucks|rocks|amazing|terrible|awful|great|good|bad)',
        ]
    
    async def analyze_aspects(
        self, 
        text: str, 
        config: AnalysisConfig
    ) -> Dict[str, float]:
        """
        Analyze sentiment towards different aspects in text
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            Dictionary mapping aspect names to sentiment scores
        """
        try:
            # Extract aspects from text
            aspects = self._extract_aspects(text)
            
            if not aspects:
                return {}
            
            # Analyze sentiment for each aspect
            aspect_sentiments = {}
            
            for aspect, contexts in aspects.items():
                # Combine all contexts for this aspect
                combined_context = " ".join(contexts)
                
                # Get sentiment score for the aspect context
                sentiment_score = get_sentiment_score(combined_context, subjectivity=False, verbose=False)
                
                aspect_sentiments[aspect] = float(sentiment_score)
            
            return aspect_sentiments
            
        except Exception as e:
            logger.error(f"Aspect analysis failed: {str(e)}")
            return {}
    
    def _extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        Extract aspects and their contexts from text
        
        Returns:
            Dictionary mapping aspect names to list of context sentences
        """
        aspects = defaultdict(list)
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for predefined aspects
            for aspect_category, keywords in self.common_aspects.items():
                for keyword in keywords:
                    if keyword in sentence_lower:
                        aspects[aspect_category].append(sentence)
                        break
            
            # Extract aspects using patterns
            extracted_aspects = self._extract_aspects_with_patterns(sentence)
            for aspect in extracted_aspects:
                if aspect not in aspects:
                    aspects[aspect].append(sentence)
        
        return dict(aspects)
    
    def _extract_aspects_with_patterns(self, sentence: str) -> List[str]:
        """Extract aspects using regex patterns"""
        aspects = []
        
        for pattern in self.aspect_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern captured multiple groups, take the first one as aspect
                    aspect = match[0]
                else:
                    aspect = match
                
                # Filter out common words that are not real aspects
                if self._is_valid_aspect(aspect):
                    aspects.append(aspect.lower())
        
        return aspects
    
    def _is_valid_aspect(self, word: str) -> bool:
        """Check if a word is a valid aspect (not a common word)"""
        # Filter out common words that are not aspects
        common_words = {
            'the', 'this', 'that', 'and', 'or', 'but', 'is', 'was', 'are', 'were',
            'very', 'really', 'quite', 'extremely', 'so', 'too', 'much', 'many',
            'good', 'bad', 'great', 'terrible', 'awful', 'amazing', 'excellent',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        return (len(word) > 2 and 
                word.lower() not in common_words and 
                word.isalpha())
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def get_aspect_summary(self, aspect_sentiments: Dict[str, float]) -> str:
        """Get a human-readable summary of aspect sentiments"""
        if not aspect_sentiments:
            return "No specific aspects detected"
        
        positive_aspects = []
        negative_aspects = []
        neutral_aspects = []
        
        for aspect, sentiment in aspect_sentiments.items():
            if sentiment > 0.1:
                positive_aspects.append(f"{aspect} (+{sentiment:.2f})")
            elif sentiment < -0.1:
                negative_aspects.append(f"{aspect} ({sentiment:.2f})")
            else:
                neutral_aspects.append(f"{aspect} (neutral)")
        
        summary_parts = []
        
        if positive_aspects:
            summary_parts.append(f"Positive: {', '.join(positive_aspects)}")
        
        if negative_aspects:
            summary_parts.append(f"Negative: {', '.join(negative_aspects)}")
        
        if neutral_aspects:
            summary_parts.append(f"Neutral: {', '.join(neutral_aspects)}")
        
        return "; ".join(summary_parts) if summary_parts else "No clear aspect sentiments"
    
    def get_dominant_aspects(self, aspect_sentiments: Dict[str, float], top_n: int = 3) -> List[Tuple[str, float]]:
        """Get the most significant aspects by absolute sentiment score"""
        if not aspect_sentiments:
            return []
        
        # Sort by absolute sentiment score
        sorted_aspects = sorted(
            aspect_sentiments.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_aspects[:top_n]
    
    async def batch_analyze_aspects(
        self, 
        texts: List[str], 
        config: AnalysisConfig
    ) -> List[Dict[str, float]]:
        """
        Analyze aspects for multiple texts
        
        Args:
            texts: List of texts to analyze
            config: Analysis configuration
            
        Returns:
            List of aspect sentiment dictionaries
        """
        results = []
        
        for text in texts:
            aspect_sentiments = await self.analyze_aspects(text, config)
            results.append(aspect_sentiments)
        
        return results
    
    def merge_aspect_results(self, results: List[Dict[str, float]]) -> Dict[str, List[float]]:
        """
        Merge aspect results from multiple texts
        
        Args:
            results: List of aspect sentiment dictionaries
            
        Returns:
            Dictionary mapping aspects to lists of sentiment scores
        """
        merged = defaultdict(list)
        
        for result in results:
            for aspect, sentiment in result.items():
                merged[aspect].append(sentiment)
        
        return dict(merged)
    
    def calculate_aspect_statistics(self, merged_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for aspects across multiple texts
        
        Args:
            merged_results: Output from merge_aspect_results
            
        Returns:
            Dictionary with statistics for each aspect
        """
        statistics = {}
        
        for aspect, scores in merged_results.items():
            if scores:
                statistics[aspect] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                    "positive_mentions": sum(1 for s in scores if s > 0.1),
                    "negative_mentions": sum(1 for s in scores if s < -0.1),
                    "neutral_mentions": sum(1 for s in scores if -0.1 <= s <= 0.1)
                }
        
        return statistics
